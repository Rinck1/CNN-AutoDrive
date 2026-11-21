/* ======================================================================
   MiniFlows Web Studio —— 3D 场景引擎（three.js）
   ======================================================================
   三种可切换的 3D 视图：
     arch      架构拆解：逐层展开，每层一个体块 + 连线 + 中文标签
     weights   3D 参数：权重张量的三维点云，按数值着色
     backward  反向传播：计算图 + 沿边回流的梯度粒子
   全部使用平滑插值动画，相机带阻尼轨道控制与自动旋转。
   ====================================================================== */
(function (global) {
  'use strict';

  var THREE = global.THREE;
  if (!THREE) { console.error('three.js 未加载，3D 视图不可用'); return; }

  /* ------------------------------------------------------------------
     工具
     ------------------------------------------------------------------ */
  function cssv(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  /* three.js 的 Color 无法解析 oklch()，而页面主题大量使用 oklch；
     因此 3D 场景使用独立的一套显式 RGB 调色板（随明暗主题切换）。 */
  var PALETTE = {
    light: {
      surface: '#ffffff', surface2: '#f3f3f5', ink: '#030213', ink2: '#3f3f46',
      mute: '#717182', faint: '#a1a1aa', line2: '#c9c9d1',
      c1: '#030213', c2: '#c2681c', c3: '#2f8fa3', c4: '#c0362b', c5: '#5a4bb5',
    },
    dark: {
      surface: '#0e0e0f', surface2: '#1c1c1f', ink: '#f7f7f8', ink2: '#d1d1d6',
      mute: '#9b9ba5', faint: '#77777f', line2: '#4a4a50',
      c1: '#f2f2f3', c2: '#f0a030', c3: '#4ec5d8', c4: '#e0564a', c5: '#8b7cf0',
    },
  };
  function themeName() { return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'; }
  function pal(name) { return PALETTE[themeName()][name] || '#888888'; }

  /* 主题化视口背景：浅色下模拟白纸结构图，深色下保持低对比 */
  function sceneBgTexture() {
    var dark = themeName() === 'dark';
    var cv = document.createElement('canvas');
    cv.width = 8; cv.height = 256;
    var g = cv.getContext('2d');
    var grd = g.createLinearGradient(0, 0, 0, 256);
    if (dark) {
      grd.addColorStop(0, '#111216');
      grd.addColorStop(0.55, '#15171c');
      grd.addColorStop(1, '#1b1e25');
    } else {
      grd.addColorStop(0, '#ffffff');
      grd.addColorStop(0.55, '#fbfbfc');
      grd.addColorStop(1, '#eef0f4');
    }
    g.fillStyle = grd; g.fillRect(0, 0, 8, 256);
    return new THREE.CanvasTexture(cv);
  }
  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }

  /* 数值 → 颜色：负值偏青、正值偏橙（与项目 c3/c2 呼应），|v| 越大越亮 */
  function valueColor(v, maxAbs, neg, pos) {
    var t = maxAbs > 0 ? clamp(Math.abs(v) / maxAbs, 0, 1) : 0;
    var c = new THREE.Color();
    var target = v >= 0 ? pos : neg;
    c.copy(new THREE.Color(neg)).lerp(new THREE.Color(pos), v >= 0 ? t : 1 - t);
    c.lerp(target, 0.15);
    var k = 0.45 + 0.55 * t;
    c.multiplyScalar(k);
    return c;
  }

  function roundSpriteTexture() {
    var cv = document.createElement('canvas');
    cv.width = cv.height = 64;
    var g = cv.getContext('2d');
    var grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.35, 'rgba(255,255,255,0.85)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    var tex = new THREE.CanvasTexture(cv);
    return tex;
  }

  /* 标签放置：按优先级贪心避让，避免互相遮挡 */
  function placeLabels(items, camera, canvas) {
    var v = new THREE.Vector3(), rect = canvas.getBoundingClientRect();
    var placed = [];
    var order = items.map(function (_, i) { return i; });
    order.sort(function (a, b) { return items[a].prio - items[b].prio; });
    order.forEach(function (i) {
      var it = items[i], el = it.el;
      v.copy(it.pos).project(camera);
      if (v.z > 1) { el.style.opacity = '0'; el.style.pointerEvents = 'none'; return; }
      var w = el.offsetWidth || 90, h = el.offsetHeight || 22;
      var px = (v.x * 0.5 + 0.5) * rect.width, py = (-v.y * 0.5 + 0.5) * rect.height;
      var box = { l: px - w / 2, r: px + w / 2, t: py - h, b: py };
      var hit = false;
      for (var q = 0; q < placed.length; q++) {
        var p = placed[q];
        if (box.l < p.r + 3 && box.r > p.l - 3 && box.t < p.b + 3 && box.b > p.t - 3) { hit = true; break; }
      }
      if (hit && !it.force) { el.style.opacity = '0'; el.style.pointerEvents = 'none'; return; }
      placed.push(box);
      el.style.opacity = String(it.opacity);
      el.style.pointerEvents = it.force ? 'auto' : 'none';
      el.style.transform = 'translate(-50%,-100%) translate(' + px.toFixed(1) + 'px,' + py.toFixed(1) + 'px)';
    });
  }

  /* 层的语义优先级：数字越小越优先获得标签位置 */
  var KIND_PRIO = {
    conv: 0, linear: 0, block: 1, attn: 1, embed: 1, model: 1,
    norm: 2, other: 2, pool: 3, act: 3, dropout: 4, flatten: 4, seq: 4,
  };
  function prioOf(kind) { return KIND_PRIO[kind] != null ? KIND_PRIO[kind] : 2; }

  /* ------------------------------------------------------------------
     轨道相机控制（自写，带阻尼与自动旋转）
     ------------------------------------------------------------------ */
  function Orbit(camera, dom, opts) {
    opts = opts || {};
    this.cam = camera; this.dom = dom;
    this.target = new THREE.Vector3(0, 0, 0);
    this.theta = opts.theta != null ? opts.theta : Math.PI * 0.13;
    this.phi = opts.phi != null ? opts.phi : 1.30;
    this.radius = opts.radius != null ? opts.radius : 20;
    this.tTheta = this.theta; this.tPhi = this.phi; this.tRadius = this.radius;
    this.autoRotate = opts.autoRotate !== false;
    this.autoSpeed = 0.22;
    this.minRadius = opts.minRadius || 3;
    this.maxRadius = opts.maxRadius || 160;
    this.enabled = true;
    this._dragging = 0; this._lx = 0; this._ly = 0;
    this._bind();
    this.update(0);
  }
  Orbit.prototype._bind = function () {
    var self = this, el = this.dom;
    el.addEventListener('pointerdown', function (e) {
      if (!self.enabled) return;
      self._dragging = e.button === 2 ? 2 : 1;
      self._lx = e.clientX; self._ly = e.clientY;
      el.setPointerCapture(e.pointerId);
      self.autoRotate = false;
    });
    el.addEventListener('pointermove', function (e) {
      if (!self._dragging) return;
      var dx = e.clientX - self._lx, dy = e.clientY - self._ly;
      self._lx = e.clientX; self._ly = e.clientY;
      if (self._dragging === 1) {
        self.tTheta -= dx * 0.006;
        self.tPhi = clamp(self.tPhi - dy * 0.006, 0.08, Math.PI - 0.08);
      } else {
        var right = new THREE.Vector3().setFromMatrixColumn(self.cam.matrix, 0);
        var up = new THREE.Vector3().setFromMatrixColumn(self.cam.matrix, 1);
        var k = self.radius * 0.0016;
        self.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
      }
    });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (ev) {
      el.addEventListener(ev, function () { self._dragging = 0; });
    });
    el.addEventListener('wheel', function (e) {
      if (!self.enabled) return;
      e.preventDefault();
      self.tRadius = clamp(self.tRadius * (1 + Math.sign(e.deltaY) * 0.12), self.minRadius, self.maxRadius);
    }, { passive: false });
    el.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  };
  Orbit.prototype.update = function (dt) {
    if (this.autoRotate && !this._dragging) this.tTheta += this.autoSpeed * dt;
    var k = 1 - Math.pow(0.0015, dt);
    this.theta += (this.tTheta - this.theta) * k;
    this.phi += (this.tPhi - this.phi) * k;
    this.radius += (this.tRadius - this.radius) * k;
    var sp = Math.sin(this.phi), cp = Math.cos(this.phi);
    this.cam.position.set(
      this.target.x + this.radius * sp * Math.sin(this.theta),
      this.target.y + this.radius * cp,
      this.target.z + this.radius * sp * Math.cos(this.theta)
    );
    this.cam.lookAt(this.target);
  };
  Orbit.prototype.frame = function (radius) {
    this.tRadius = radius; this.tTheta = Math.PI * 0.28; this.tPhi = 1.15;
    this.target.set(0, 0, 0);
  };

  /* ------------------------------------------------------------------
     场景基类
     ------------------------------------------------------------------ */
  function BaseScene(canvas, overlay) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.clock = new THREE.Clock();
    this.disposed = false;

    this.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    /* 卡片是纯色/贴图材质，保持 sRGB 原色，不做电影感色调映射 */
    if (THREE.NoToneMapping !== undefined) this.renderer.toneMapping = THREE.NoToneMapping;
    /* 真实 3D 立体感：阴影贴图 */
    this.renderer.shadowMap.enabled = true;
    if (THREE.PCFSoftShadowMap !== undefined) this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 900);
    this.orbit = new Orbit(this.camera, canvas, { radius: 22 });

    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.dir = new THREE.DirectionalLight(0xffffff, 1.05);
    this.dir.position.set(14, 20, 12);
    this.dir.castShadow = true;
    this.dir.shadow.mapSize.width = 1024;
    this.dir.shadow.mapSize.height = 1024;
    var sc = this.dir.shadow.camera;
    sc.left = -70; sc.right = 70; sc.top = 70; sc.bottom = -70;
    sc.near = 1; sc.far = 240;
    this.dir.shadow.bias = -0.0007;
    this.hemi = new THREE.HemisphereLight(0xcfe0ff, 0x1a2230, 0.55);
    this.fill = new THREE.DirectionalLight(0x9ec4ff, 0.35);
    this.fill.position.set(-10, 5, -12);
    this.scene.add(this.ambient, this.dir, this.hemi, this.fill);
    this.scene.add(this.dir.target);

    this.applyTheme();
    this._resize = this._resize.bind(this);
    window.addEventListener('resize', this._resize);
    var self = this;
    this._ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(function () { self._resize(); }) : null;
    if (this._ro) this._ro.observe(canvas);
    this._resize();
    this._loop = this._loop.bind(this);
    this.raf = requestAnimationFrame(this._loop);
  }
  BaseScene.prototype.applyTheme = function () {
    var dark = themeName() === 'dark';
    this.scene.background = new THREE.Color(dark ? 0x14161b : 0xffffff);
    this.renderer.setClearColor(dark ? 0x14161b : 0xffffff, 1);
    this.ambient.color = new THREE.Color(0xffffff);
    this.ambient.intensity = dark ? 0.55 : 0.95;
    this.dir.color = new THREE.Color(0xffffff);
    this.dir.intensity = dark ? 0.95 : 0.55;
    this.hemi.color = new THREE.Color(dark ? 0xcfe0ff : 0xffffff);
    this.hemi.groundColor = new THREE.Color(dark ? 0x1a2230 : 0xe8e8ee);
    this.hemi.intensity = dark ? 0.55 : 0.35;
    this.scene.fog = null;
    if (this.onTheme) this.onTheme();
  };
  BaseScene.prototype._resize = function () {
    var r = this.canvas.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) return;
    var w = Math.round(r.width), h = Math.round(r.height);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  };
  BaseScene.prototype._loop = function () {
    if (this.disposed) return;
    var dt = Math.min(0.05, this.clock.getDelta());
    this.dt = dt;
    this.update(dt);
    this.render();
    this.raf = requestAnimationFrame(this._loop);
  };
  BaseScene.prototype.update = function (dt) { this.orbit.update(dt); };
  BaseScene.prototype.resetView = function () {
    if (this.viewRadius) this.orbit.frame(this.viewRadius);
    if (this.viewCenter) this.orbit.target.copy(this.viewCenter);
    if (this.viewTheta != null) this.orbit.tTheta = this.viewTheta;
    if (this.viewPhi != null) this.orbit.tPhi = this.viewPhi;
  };
  BaseScene.prototype.render = function () {
    this.renderer.render(this.scene, this.camera);
    this.syncLabels();
  };
  BaseScene.prototype.syncLabels = function () { };
  BaseScene.prototype.clearGroup = function () {
    while (this.group.children.length) {
      var c = this.group.children[0];
      this.group.remove(c);
      if (c.geometry) c.geometry.dispose();
      if (c.material) {
        if (Array.isArray(c.material)) c.material.forEach(function (m) { m.dispose(); });
        else c.material.dispose();
      }
    }
  };
  BaseScene.prototype.dispose = function () {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this._resize);
    if (this._ro) { try { this._ro.disconnect(); } catch (e) { } }
    this.clearGroup();
    if (this.overlay) this.overlay.innerHTML = '';
    try { this.renderer.dispose(); } catch (e) { }
  };

  /* ------------------------------------------------------------------
     视图一：架构拆解（矩阵 / 张量网络示意）
     每层画成一组垂直于数据流方向的「矩阵薄片」：网格纹理表达 H×W 特征图，
     通道数大于 1 时叠成多片；层与层之间的间距由真实块宽累积计算。
     ------------------------------------------------------------------ */
  var KIND_COLOR = {
    conv: 'c2', linear: 'c5', pool: 'c3', act: 'c4', norm: 'c5',
    flatten: 'c3', attn: 'c4', block: 'c1', embed: 'c2',
    dropout: 'mute', seq: 'mute', model: 'c1', other: 'c3',
  };
  var KEY_KINDS = { conv: 1, linear: 1, model: 1, block: 1, embed: 1, attn: 1 };
  function kindHex(kind) { return pal(KIND_COLOR[kind] || 'c3'); }
  function kindColor(kind) { return new THREE.Color(kindHex(kind)); }
  /* 卷积层按 stage 换色，整张图更像经典的 CNN 结构示意图 */
  var STAGE_COLORS = ['#e8a04a', '#e2703c', '#d15438', '#b8483a', '#a0433c'];
  function layerHex(L) {
    if (L.kind === 'conv') {
      var m = /stage(\d+)/.exec(L.name || '');
      var s = m ? parseInt(m[1], 10) + 1 : 0;
      return STAGE_COLORS[s % STAGE_COLORS.length];
    }
    return kindHex(L.kind);
  }
  function hexToRgb(hex) {
    var h = String(hex).replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16) || 0;
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function ArchScene(canvas, overlay) {
    BaseScene.call(this, canvas, overlay);
    this.explode = 0; this.explodeTarget = 0;
    this.selected = -1; this.hovered = -1;
    this.layers = []; this.labels = []; this.meshes = []; this.groups = [];
    this._actTex = []; this._texCache = {};
    this.spriteTex = roundSpriteTexture();
    this.maxH = 4;
    this.disp = []; this.arrows = []; this.inputLabel = null;
    this.inputImage = null;
    /* 地面 + 网格：给 3D 结构图提供空间参照和阴影承接面 */
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(260, 260),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1.0, metalness: 0.0 })
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -3.2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    this.grid = new THREE.GridHelper(260, 60, 0xc8ced8, 0xe4e7ed);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.28;
    this.grid.position.y = -3.16;
    this.scene.add(this.grid);
  }
  ArchScene.prototype = Object.create(BaseScene.prototype);
  ArchScene.prototype.constructor = ArchScene;
  ArchScene.prototype.onTheme = function () {
    var dark = themeName() === 'dark';
    if (this.ground) this.ground.material.color.set(dark ? 0x15171c : 0xffffff);
    if (this.grid) this.grid.material.opacity = dark ? 0.26 : 0.48;
  };

  /* 没有真实激活数据时，合成一个按层序号变化的占位矩阵 */
  ArchScene.prototype._synthThumb = function (rows, cols, seed) {
    var nr = clamp(Math.round(rows || 4), 1, 12);
    var nc = clamp(Math.round(cols || 4), 1, 12);
    var data = [];
    for (var r = 0; r < nr; r++) {
      var row = [];
      for (var c = 0; c < nc; c++) {
        var v = 0.5 + 0.5 * Math.sin((r + 1.3) * (c + 2.1) + seed * 0.7);
        row.push(+(v * 0.75).toFixed(3));
      }
      data.push(row);
    }
    return { shape: [nr, nc], data: data };
  };

  /* 把真实激活热力图绘制成纹理：类别色卡片 + 明暗纹理（参考结构图风格） */
  ArchScene.prototype._activationTexture = function (thumb, hex) {
    var rows = Math.max(1, Math.round((thumb.shape && thumb.shape[0]) || 1));
    var cols = Math.max(1, Math.round((thumb.shape && thumb.shape[1]) || 1));
    var data = thumb.data || [];
    var S = 192, pad = 4;
    var cv = document.createElement('canvas');
    cv.width = cv.height = S;
    var g = cv.getContext('2d');
    var rgb = hexToRgb(hex);
    /* 卡片底色：类别色向白色提亮一点，避免过暗 */
    var br = Math.round(rgb[0] + (255 - rgb[0]) * 0.05);
    var bg = Math.round(rgb[1] + (255 - rgb[1]) * 0.05);
    var bb = Math.round(rgb[2] + (255 - rgb[2]) * 0.05);
    g.fillStyle = 'rgb(' + br + ',' + bg + ',' + bb + ')';
    g.fillRect(0, 0, S, S);
    var cw = (S - pad * 2) / cols, ch = (S - pad * 2) / rows;
    for (var r = 0; r < rows; r++) {
      var line = data[r] || [];
      for (var c = 0; c < cols; c++) {
        var v = clamp(+line[c] || 0, 0, 1);
        var t = 0.80 + 0.30 * Math.pow(v, 0.70);
        var R = clamp(Math.round(br * t), 0, 255);
        var G = clamp(Math.round(bg * t), 0, 255);
        var B = clamp(Math.round(bb * t), 0, 255);
        g.fillStyle = 'rgb(' + R + ',' + G + ',' + B + ')';
        g.fillRect(pad + c * cw + 0.4, pad + r * ch + 0.4, Math.max(1, cw - 0.8), Math.max(1, ch - 0.8));
      }
    }
    g.strokeStyle = 'rgba(255,255,255,0.10)';
    g.lineWidth = 1;
    for (var gi = 1; gi < rows; gi++) { g.beginPath(); g.moveTo(pad, pad + gi * ch); g.lineTo(S - pad, pad + gi * ch); g.stroke(); }
    for (var gj = 1; gj < cols; gj++) { g.beginPath(); g.moveTo(pad + gj * cw, pad); g.lineTo(pad + gj * cw, S - pad); g.stroke(); }
    g.strokeStyle = 'rgba(0,0,0,0.18)';
    g.lineWidth = 3;
    g.strokeRect(1.5, 1.5, S - 3, S - 3);
    var tex = new THREE.CanvasTexture(cv);
    tex.minFilter = THREE.LinearFilter;
    this._actTex.push(tex);
    return tex;
  };

  /* 输入图像纹理（灰度） */
  ArchScene.prototype._imageTexture = function (img) {
    var rows = img.length, cols = (img[0] || []).length || 1;
    var S = 160, cv = document.createElement('canvas');
    cv.width = cv.height = S;
    var g = cv.getContext('2d');
    var cw = S / cols, ch = S / rows;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var v = clamp(+((img[r] || [])[c]) || 0, 0, 1);
        var gv = Math.round(255 - v * 225);
        g.fillStyle = 'rgb(' + gv + ',' + gv + ',' + gv + ')';
        g.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);
      }
    }
    g.strokeStyle = 'rgba(0,0,0,0.18)';
    g.lineWidth = 2;
    g.strokeRect(1, 1, S - 2, S - 2);
    var tex = new THREE.CanvasTexture(cv);
    tex.minFilter = THREE.LinearFilter;
    this._actTex.push(tex);
    return tex;
  };

  ArchScene.prototype._clearActTex = function () {
    for (var i = 0; i < this._actTex.length; i++) {
      try { this._actTex[i].dispose(); } catch (e) { }
    }
    this._actTex = [];
  };

  function layerShapeText(L) {
    var s = L.out_shape || [];
    if (s.length >= 4) return s[s.length - 1] + '×' + s[s.length - 2] + '×' + s[s.length - 3];
    if (s.length === 3) return s[s.length - 1] + '×' + s[s.length - 2];
    if (s.length === 2) return String(s[s.length - 1]);
    return L.type || '';
  }

  /* 把逐层清单压缩成主干阶段：stageN.blockM 归为一个块 */
  function blockKey(name) {
    var m = /^(.*?block\d+)/.exec(name || '');
    if (m) return m[1];
    var parts = (name || '').split('.');
    if (parts.length > 1 && parts[0] === 'stem') return 'stem';
    return name || 'layer';
  }

  ArchScene.prototype.setData = function (data) {
    this.clearGroup();
    this.overlay.innerHTML = '';
    this._clearActTex();
    this.meshes = []; this.labels = []; this.groups = []; this.arrows = [];
    this.inputLabel = null; this.inputLabelPos = null; this.inputArrow = null;
    this.layers = (data.layout && data.layout.layers) || [];
    this.inputImage = data.input_image || null;
    if (!this.layers.length) return;
    var self = this;

    /* 按 block 分组：只画主干阶段（stem / stageN.blockM / head），每块用最深一层的真实特征图 */
    var DISP = { conv: 1, pool: 1, linear: 1, embed: 1, attn: 1, model: 1, flatten: 1 };
    var buckets = {}, order = [];
    this.layers.forEach(function (L) {
      if (!DISP[L.kind]) return;
      var key = blockKey(L.name || L.type);
      if (!buckets[key]) { buckets[key] = []; order.push(key); }
      buckets[key].push(L);
    });
    var disp = order.map(function (key) {
      var arr = buckets[key];
      var mains = arr.filter(function (L) { return !/shortcut/i.test(L.name || ''); });
      var rep = (mains.length ? mains : arr).slice(-1)[0];
      rep._group = key;
      return rep;
    });
    if (disp.length < 3) disp = this.layers.slice();
    this.disp = disp;

    /* 单行长廊布局：沿 X 轴排布，块的通道厚度就是数据流方向进深 */
    var maxVW = 0, maxVH = 0;
    disp.forEach(function (L) {
      maxVW = Math.max(maxVW, L.vw || L.d || 1.5);
      maxVH = Math.max(maxVH, L.vh || L.h || 1.5);
    });
    var cursor = 0;
    disp.forEach(function (L, i) {
      if (i > 0) {
        var p = disp[i - 1];
        var gap = 0.85 + Math.max(p.vw || 1.5, L.vw || 1.5) * 0.10;
        cursor += (p.thick || 0.4) / 2 + gap + (L.thick || 0.4) / 2;
      }
      L._x = cursor; L._y = 0; L._row = 0;
    });
    var xMid = (disp[0]._x + disp[disp.length - 1]._x) / 2;
    disp.forEach(function (L) { L._x -= xMid; });
    this.maxVH = maxVH;
    this.arrowY = -(maxVH / 2) - 0.55;

    /* 输入图像卡片 */
    var inputX = null, inputY = 0;
    var inputSize = Math.min(3.9, Math.max(2.4, (disp[0].vw || 2.5) * 1.05));
    if (this.inputImage) {
      var first = disp[0];
      inputX = first._x - (first.thick || first.w) / 2 - 1.0 - inputSize / 2;
      var imgTex = this._imageTexture(this.inputImage);
      var imgPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(inputSize, inputSize),
        new THREE.MeshBasicMaterial({ map: imgTex, side: THREE.DoubleSide, transparent: true, opacity: 0.98 })
      );
      imgPlane.position.set(inputX, inputY, 0.03);
      this.group.add(imgPlane);
      var imgFrame = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.PlaneGeometry(inputSize, inputSize)),
        new THREE.LineBasicMaterial({ color: pal('ink2'), transparent: true, opacity: 0.45 })
      );
      imgFrame.position.copy(imgPlane.position);
      this.group.add(imgFrame);
      this.inputLabel = document.createElement('div');
      this.inputLabel.className = 'm3d-label tiny';
      this.inputLabel.innerHTML = '<b>输入图像</b><span>' +
        (this.inputImage[0] ? this.inputImage[0].length : 0) + '×' + this.inputImage.length + '</span>';
      this.overlay.appendChild(this.inputLabel);
      this.inputLabelPos = new THREE.Vector3(inputX, inputY + inputSize / 2 + 0.35, 0);
    }

    /* 每一层：一叠特征图卡片（XY 平面，沿数据流方向叠出通道厚度） */
    disp.forEach(function (L, di) {
      var hex = layerHex(L);
      var base = new THREE.Color(hex);
      var vw = L.vw || L.d || 1.5, vh = L.vh || L.h || 1.5;
      var thick = L.thick || L.w || 0.4;
      var thumb = (L.thumb && L.thumb.shape) ? L.thumb : self._synthThumb(L.rows, L.cols, di);
      var tex = self._activationTexture(thumb, hex);
      var nCards = clamp(Math.round(1 + Math.log2(Math.max(1, L.channels || 1))), 1, 7);
      var gap = nCards > 1 ? (thick * 0.86) / (nCards - 1) : 0;

      var grp = new THREE.Group();
      grp.position.set(L._x, L._y, 0);
      var cards = [];

      /* 实体背板：给整叠特征图一个真实体积，并投射阴影 */
      var body = new THREE.Mesh(
        new THREE.BoxGeometry(thick + 0.05, vh + 0.04, 0.55),
        new THREE.MeshStandardMaterial({
          color: base, roughness: 0.45, metalness: 0.18,
          emissive: base, emissiveIntensity: 0.10,
          transparent: true, opacity: 0.96,
        })
      );
      body.position.z = -0.30;
      body.castShadow = true;
      body.receiveShadow = true;
      grp.add(body);

      for (var k = 0; k < nCards; k++) {
        var mat = new THREE.MeshBasicMaterial({
          map: tex, transparent: true, opacity: 0.98,
          side: THREE.DoubleSide, depthWrite: false,
        });
        var shade = nCards > 1 ? (0.72 + 0.28 * (k / (nCards - 1))) : 1.0;
        mat.color.setScalar(shade);
        var card = new THREE.Mesh(new THREE.PlaneGeometry(vw, vh), mat);
        card.position.set((k - (nCards - 1) / 2) * gap, 0, 0.002);
        card.userData.shade = shade;
        grp.add(card);
        cards.push(card);
      }

      /* 整叠卡片的边框 */
      var rim = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(thick + 0.06, vh + 0.05, 0.60)),
        new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.72 })
      );
      rim.position.z = -0.30;
      grp.add(rim);

      /* 透明拾取盒 */
      var pick = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.5, thick + 0.05), vh, 0.70),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false })
      );
      pick.position.z = -0.30;
      pick.userData = { idx: L.idx, name: L.name, kind: L.kind };
      grp.add(pick);

      self.group.add(grp);
      self.groups.push({ grp: grp, cards: cards, rim: rim, pick: pick, body: body, layer: L });
      self.meshes.push(pick);

      var el = document.createElement('div');
      el.className = 'm3d-label tiny';
      el.innerHTML = '<b>' + (L._group || L.name) + '</b><span>' + layerShapeText(L) + '</span>';
      self.overlay.appendChild(el);
      self.labels.push(el);
    });

    /* 层与层之间的箭头 */
    for (var ai = 1; ai < this.groups.length; ai++) self._makeArrow(ai - 1, ai);
    if (inputX !== null && this.groups.length) {
      var firstL = disp[0];
      var x0 = inputX + inputSize / 2 + 0.10;
      var x1 = firstL._x - (firstL.thick || firstL.w) / 2 - 0.22;
      this.inputArrow = self._makeArrowLine(new THREE.Vector3(x0, this.arrowY, 0.20),
                                            new THREE.Vector3(x1, this.arrowY, 0.20));
    }

    /* 相机：等距 3/4 视角，同时看到正面、顶面与进深 */
    var minX = Infinity, maxX = -Infinity;
    disp.forEach(function (L) {
      var half = (L.thick || 0.4) / 2 + 0.12;
      minX = Math.min(minX, L._x - half);
      maxX = Math.max(maxX, L._x + half);
    });
    if (inputX !== null) minX = Math.min(minX, inputX - inputSize / 2);
    var spanX = Math.max(8, maxX - minX);
    this.span = spanX;
    var radius = Math.max(10, 1.00 * spanX);
    this.orbit.frame(radius);
    this.orbit.target.set((minX + maxX) / 2, 0.25, 0);
    this.orbit.tTheta = Math.PI * 0.26;
    this.orbit.tPhi = 1.20;
    this.viewRadius = radius;
    this.viewCenter = new THREE.Vector3((minX + maxX) / 2, 0.25, 0);
    this.viewTheta = Math.PI * 0.26;
    this.viewPhi = 1.20;

    /* 地面高度、阴影光源跟随整体 */
    this.ground.position.y = -(maxVH / 2) - 0.55;
    this.grid.position.y = this.ground.position.y + 0.02;
    this.dir.target.position.set((minX + maxX) / 2, 0, 0);
    this.dir.position.set((minX + maxX) / 2 + 14, 20, 12);

    this.select(-1);
  };

  ArchScene.prototype._makeArrowLine = function (p0, p1) {
    var geo = new THREE.BufferGeometry().setFromPoints([p0, p1]);
    var line = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: pal('ink2'), transparent: true, opacity: 0.60,
    }));
    var cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.10, 0.26, 12),
      new THREE.MeshBasicMaterial({ color: pal('ink2') })
    );
    cone.rotation.z = -Math.PI / 2;
    cone.position.set(p1.x, p1.y, p1.z);
    this.group.add(line); this.group.add(cone);
    return { line: line, cone: cone };
  };

  ArchScene.prototype._makeArrow = function (a, b) {
    var arrow = this._makeArrowLine(new THREE.Vector3(0, 0, -0.04),
                                    new THREE.Vector3(0.2, 0, -0.04));
    arrow.a = a; arrow.b = b;
    this.arrows.push(arrow);
    return arrow;
  };

  ArchScene.prototype.setExplode = function (v) { this.explodeTarget = clamp(v, 0, 1); };

  ArchScene.prototype._applyStyle = function () {
    var sel = this.selected, hov = this.hovered;
    for (var di = 0; di < this.groups.length; di++) {
      var g = this.groups[di], L = this.disp[di];
      var on = (sel < 0 || L.idx === sel);
      var focus = (L.idx === sel || L.idx === hov);
      for (var k = 0; k < g.cards.length; k++) {
        g.cards[k].material.opacity = on ? (focus ? 1.0 : 0.97) : 0.30;
      }
      g.body.material.opacity = on ? (focus ? 1.0 : 0.94) : 0.16;
      g.body.material.emissive.set(focus ? pal('c2') : layerHex(L));
      g.body.material.emissiveIntensity = focus ? 0.55 : 0.10;
      g.rim.material.color.set(focus ? pal('c2') : layerHex(L));
      g.rim.material.opacity = focus ? 1.0 : (on ? 0.62 : 0.14);
      g.grp.scale.setScalar(focus ? 1.04 : 1.0);
    }
    for (var j = 0; j < this.labels.length; j++) {
      this.labels[j].classList.toggle('sel', this.disp[j].idx === sel);
    }
  };

  ArchScene.prototype.select = function (idx) {
    this.selected = idx;
    this._applyStyle();
  };

  ArchScene.prototype.hover = function (idx) {
    if (this.hovered === idx) return;
    this.hovered = idx;
    this._applyStyle();
    this.canvas.style.cursor = idx >= 0 ? 'pointer' : 'grab';
  };

  ArchScene.prototype.update = function (dt) {
    this.explode += (this.explodeTarget - this.explode) * (1 - Math.pow(0.004, dt));
    var e = ease(this.explode);
    for (var di = 0; di < this.groups.length; di++) {
      var L = this.disp[di], g = this.groups[di];
      g.grp.position.x = L._x * (1 + e * 0.32);
      g.grp.position.y = L._y + Math.sin(di * 0.8) * e * 0.42;
      g.grp.position.z = Math.cos(di * 0.6) * e * 0.30;
    }
    /* 箭头跟随块位置重算（位于块下方，像一条数据流水线） */
    for (var i = 0; i < this.arrows.length; i++) {
      var ar = this.arrows[i];
      var A = this.groups[ar.a].grp.position, B = this.groups[ar.b].grp.position;
      var la = this.disp[ar.a], lb = this.disp[ar.b];
      var x0 = A.x + (la.thick || la.w || 0.4) / 2 + 0.06;
      var x1 = B.x - (lb.thick || lb.w || 0.4) / 2 - 0.22;
      var y = this.arrowY;
      var arr = ar.line.geometry.attributes.position;
      arr.setXYZ(0, x0, y, 0.20);
      arr.setXYZ(1, x1, y, 0.20);
      arr.needsUpdate = true;
      ar.cone.position.set(x1, y, 0.20);
      ar.cone.rotation.z = -Math.PI / 2;
    }
    if (this.inputArrow && this.groups.length) {
      var fg = this.groups[0].grp.position, fl = this.disp[0];
      var ia = this.inputArrow.line.geometry.attributes.position;
      var ex = fg.x - (fl.thick || fl.w || 0.4) / 2 - 0.22;
      ia.setXYZ(1, ex, this.arrowY, 0.20);
      ia.needsUpdate = true;
      this.inputArrow.cone.position.set(ex, this.arrowY, 0.20);
      this.inputArrow.cone.rotation.z = -Math.PI / 2;
    }
    this.orbit.update(dt);
  };

  ArchScene.prototype.syncLabels = function () {
    if (!this.labels.length) return;
    var items = [], v = new THREE.Vector3();
    var sel = this.selected, hov = this.hovered;
    for (var di = 0; di < this.labels.length; di++) {
      var L = this.disp[di], el = this.labels[di];
      if (!L || !el) continue;
      var focus = (L.idx === sel || L.idx === hov);
      if (sel >= 0 && !focus) { el.style.opacity = '0'; el.style.pointerEvents = 'none'; continue; }
      v.set(0, +((L.vh || L.h) / 2 + 0.35), 0);
      this.groups[di].grp.localToWorld(v);
      items.push({
        el: el, pos: v.clone(),
        prio: focus ? -1 : prioOf(L.kind),
        force: focus,
        opacity: focus ? 1 : 0.9,
      });
    }
    if (this.inputLabel && this.inputLabelPos && sel < 0) {
      items.push({ el: this.inputLabel, pos: this.inputLabelPos.clone(), prio: -2, force: false, opacity: 0.9 });
    }
    placeLabels(items, this.camera, this.canvas);
  };

  ArchScene.prototype.pick = function (clientX, clientY) {
    var rect = this.canvas.getBoundingClientRect();
    var mx = ((clientX - rect.left) / rect.width) * 2 - 1;
    var my = -((clientY - rect.top) / rect.height) * 2 + 1;
    var ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(mx, my), this.camera);
    var hits = ray.intersectObjects(this.meshes, false);
    return hits.length ? hits[0].object.userData.idx : -1;
  };

  /* ------------------------------------------------------------------
     视图二：3D 参数点云
     ------------------------------------------------------------------ */
  function WeightScene(canvas, overlay) {
    BaseScene.call(this, canvas, overlay);
    this.spin = 0;
    this.groups = [];
    this.maxAbs = 1e-6;
    this.neg = new THREE.Color(pal('c3'));
    this.pos = new THREE.Color(pal('c2'));
  }
  WeightScene.prototype = Object.create(BaseScene.prototype);
  WeightScene.prototype.constructor = WeightScene;
  WeightScene.prototype.onTheme = function () {
    if (this.neg) this.neg.set(pal('c3'));
    if (this.pos) this.pos.set(pal('c2'));
  };

  WeightScene.prototype.setData = function (data) {
    this.clearGroup();
    this.overlay.innerHTML = '';
    this.groups = [];
    this.labels = [];
    var tensors = data.weights || [];
    var self = this;
    if (!tensors.length) return;

    this.maxAbs = 0;
    tensors.forEach(function (t) { t.points.forEach(function (p) { self.maxAbs = Math.max(self.maxAbs, Math.abs(p.v)); }); });

    /* 只展示参数量最大的 12 个张量，排成网格，避免 33 个点云连成一条灰线 */
    var list = tensors.slice().sort(function (a, b) { return b.total - a.total; }).slice(0, 12);
    var radii = list.map(function (t) {
      var size = Math.max(0.10, Math.min(0.55, Math.cbrt(t.total) * 0.055));
      return { size: size, r: Math.max(0.8, size * 5.2) };
    });
    var cols = Math.ceil(Math.sqrt(list.length));
    var rows = Math.ceil(list.length / cols);
    var maxR = Math.max.apply(null, radii.map(function (o) { return o.r; }));
    var cell = maxR * 2 + 2.2;

    list.forEach(function (t, ti) {
      var n = t.points.length;
      if (!n) return;
      var size = radii[ti].size;
      /* 每个张量用自身标准差着色，小权重层也能看出结构 */
      var localMax = 1e-6;
      t.points.forEach(function (p) { localMax = Math.max(localMax, Math.abs(p.v)); });
      var colorScale = Math.max(localMax * 0.30, (t.std || 0) * 2.5, 1e-6);
      var pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
      var c = new THREE.Color();
      t.points.forEach(function (p, i) {
        pos[i * 3] = p.x * size * 5.2;
        pos[i * 3 + 1] = p.y * size * 5.2;
        pos[i * 3 + 2] = p.z * size * 5.2;
        c.copy(valueColor(p.v, colorScale, self.neg, self.pos));
        col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
      });
      var geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
      var mat = new THREE.PointsMaterial({
        size: 0.18, vertexColors: true, sizeAttenuation: true,
        transparent: true, opacity: 0.95, depthWrite: false,
      });
      var pts = new THREE.Points(geo, mat);
      var colIdx = ti % cols, rowIdx = Math.floor(ti / cols);
      pts.position.set((colIdx - (cols - 1) / 2) * cell,
                       ((rows - 1) / 2 - rowIdx) * cell, 0);
      pts.userData = { name: t.name, shape: t.shape, total: t.total, r: radii[ti].r };
      self.group.add(pts);
      self.groups.push(pts);
    });

    /* 名称标签 */
    this.labels = list.map(function (t) {
      var el = document.createElement('div');
      el.className = 'm3d-label tiny';
      el.innerHTML = '<b>' + t.name.replace(/\.W$/, '') + '</b><span>' + t.shape.join('×') + ' · ' + t.total.toLocaleString() + '</span>';
      self.overlay.appendChild(el);
      return el;
    });
    var spanX = cols * cell, spanY = rows * cell;
    var radius = Math.max(20, 1.20 * Math.max(spanX / 0.96, spanY / 0.72));
    this.orbit.frame(radius);
    this.orbit.target.set(0, 0, 0);
    this.orbit.tTheta = Math.PI * 0.16;
    this.orbit.tPhi = 1.46;
    this.viewRadius = radius;
    this.viewCenter = new THREE.Vector3(0, 0, 0);
    this.viewTheta = Math.PI * 0.16;
    this.viewPhi = 1.46;
    this.spinAmt = 0;
  };
  WeightScene.prototype.update = function (dt) {
    if (this.orbit.autoRotate) this.spin += dt * 0.22;
    var want = this.orbit.autoRotate ? 1 : 0;
    this.spinAmt += (want - this.spinAmt) * (1 - Math.pow(0.005, dt));
    for (var i = 0; i < this.groups.length; i++) {
      this.groups[i].rotation.y = this.spin * (0.5 + (i % 3) * 0.22) * this.spinAmt;
      this.groups[i].rotation.x = Math.sin(this.spin * 0.6 + i) * 0.28 * this.spinAmt;
    }
    this.orbit.update(dt);
  };
  WeightScene.prototype.syncLabels = function () {
    if (!this.labels || !this.labels.length) return;
    var items = [];
    for (var i = 0; i < this.groups.length && i < this.labels.length; i++) {
      var g = this.groups[i], el = this.labels[i];
      var r = (g.userData && g.userData.r) || 1;
      items.push({
        el: el,
        pos: new THREE.Vector3(g.position.x, g.position.y - r - 0.40, g.position.z),
        prio: i % 2,
        force: false,
        opacity: 0.92,
      });
    }
    placeLabels(items, this.camera, this.canvas);
  };

  /* ------------------------------------------------------------------
     视图三：反向传播
     ------------------------------------------------------------------ */
  function BackwardScene(canvas, overlay) {
    BaseScene.call(this, canvas, overlay);
    this.groups = []; this.labels = []; this.bars = []; this.labelBars = [];
    this.wave = -1; this.waveT = 0;
  }
  BackwardScene.prototype = Object.create(BaseScene.prototype);
  BackwardScene.prototype.constructor = BackwardScene;

  /* 把计算图按 block 聚合成梯度柱：柱高 / 颜色 = 该阶段回传梯度强度 */
  BackwardScene.prototype.setData = function (data) {
    this.clearGroup();
    this.overlay.innerHTML = '';
    this.groups = []; this.labels = []; this.bars = []; this.labelBars = [];
    this.wave = -1;
    var layout = data.layout;
    if (!layout || !layout.layers || !layout.layers.length) return;

    var DISP = { conv: 1, pool: 1, linear: 1, embed: 1, attn: 1, model: 1, flatten: 1 };
    var buckets = {}, order = [];
    layout.layers.forEach(function (L) {
      if (!DISP[L.kind]) return;
      var key = blockKey(L.name || L.type);
      if (!buckets[key]) { buckets[key] = []; order.push(key); }
      buckets[key].push(L);
    });
    var disp = order.map(function (key) {
      var arr = buckets[key];
      var mains = arr.filter(function (L) { return !/shortcut/i.test(L.name || ''); });
      var rep = (mains.length ? mains : arr).slice(-1)[0];
      rep._group = key;
      return rep;
    });
    if (disp.length < 3) disp = layout.layers.slice();
    this.disp = disp;

    var self = this;
    var agg = {};
    (data.layers || []).forEach(function (g) {
      var k = blockKey(g.name || '');
      agg[k] = (agg[k] || 0) + (g.grad || 0);
    });
    var grads = disp.map(function (L) { return agg[L._group] || 0; });
    var maxG = Math.max.apply(null, grads.concat([1e-9]));

    var rows = disp.length >= 5 ? 2 : 1;
    var perRow = Math.ceil(disp.length / rows);
    var bw = 1.15, gap = 1.60, rowStep = 6.8, baseOffset = 2.6;
    var xs = [0];
    for (var i = 1; i < perRow; i++) xs.push(xs[i - 1] + bw + gap);
    disp.forEach(function (L, i) {
      var row = Math.floor(i / perRow), col = i % perRow;
      var idx = (row % 2 === 0) ? col : (perRow - 1 - col);
      L._row = row;
      L._x = xs[Math.max(0, Math.min(xs.length - 1, idx))];
      L._y = ((rows - 1) / 2 - row) * rowStep;
    });
    var xMid = (xs[0] + xs[xs.length - 1]) / 2;
    disp.forEach(function (L) { L._x -= xMid; });

    /* 每行的基线 */
    for (var r = 0; r < rows; r++) {
      var baseY = ((rows - 1) / 2 - r) * rowStep - baseOffset;
      var lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-xs[xs.length - 1] / 2 - 0.4, baseY, 0),
        new THREE.Vector3(xs[xs.length - 1] / 2 + 0.4, baseY, 0),
      ]);
      self.group.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
        color: pal('line2'), transparent: true, opacity: 0.65,
      })));
    }

    disp.forEach(function (L, i) {
      var ratio = clamp(grads[i] / maxG, 0, 1);
      var h = 0.55 + 3.9 * Math.sqrt(ratio);
      var color = new THREE.Color(pal('c3')).lerp(new THREE.Color(pal('c2')), Math.min(1, ratio * 2.2));
      if (ratio > 0.45) color.lerp(new THREE.Color(pal('c4')), (ratio - 0.45) / 0.55);
      var baseY = L._y - baseOffset;
      var mat = new THREE.MeshStandardMaterial({
        color: color, roughness: 0.34, metalness: 0.22,
        emissive: color, emissiveIntensity: 0.18,
        transparent: true, opacity: 0.94,
      });
      var bar = new THREE.Mesh(new THREE.BoxGeometry(bw, h, 1.05), mat);
      bar.position.set(L._x, baseY + h / 2, 0);
      self.group.add(bar);
      self.bars.push(bar);

      var el = document.createElement('div');
      el.className = 'm3d-label tiny';
      var gtxt = grads[i] >= 1e-4 ? grads[i].toFixed(4) : grads[i].toExponential(1);
      el.innerHTML = '<b>' + L._group + '</b><span>∇ ' + gtxt + '</span>';
      self.overlay.appendChild(el);
      self.labels.push(el);
      self.labelBars.push({ el: el, x: L._x, y: baseY - 0.34 });
    });

    var spanX = perRow * (bw + gap);
    var spanY = rows * rowStep;
    var radius = Math.max(14, 1.20 * Math.max(spanX / 0.96, spanY / 0.72));
    this.orbit.frame(radius);
    this.orbit.target.set(0, 0, 0);
    this.orbit.tTheta = Math.PI * 0.20;
    this.orbit.tPhi = 1.42;
    this.viewRadius = radius;
    this.viewCenter = new THREE.Vector3(0, 0, 0);
    this.viewTheta = Math.PI * 0.20;
    this.viewPhi = 1.42;
  };

  BackwardScene.prototype.playWave = function () { this.wave = 0; this.waveT = 0; };

  BackwardScene.prototype.update = function (dt) {
    if (this.wave >= 0 && this.bars.length) {
      this.waveT += dt * 1.5;
      var n = Math.max(1, this.bars.length - 1);
      for (var i = 0; i < this.bars.length; i++) {
        var d = Math.abs(i / n - Math.min(1, this.waveT));
        var g = Math.max(0, 1 - d * 3.2);
        this.bars[i].material.emissiveIntensity = 0.18 + g * 1.0;
      }
      if (this.waveT > 1.35) {
        this.wave = -1;
        for (var j = 0; j < this.bars.length; j++) this.bars[j].material.emissiveIntensity = 0.18;
      }
    }
    this.orbit.update(dt);
  };

  BackwardScene.prototype.syncLabels = function () {
    if (!this.labels.length) return;
    var items = [];
    for (var i = 0; i < this.labelBars.length; i++) {
      var b = this.labelBars[i];
      items.push({
        el: b.el,
        pos: new THREE.Vector3(b.x, b.y, 0),
        prio: 0, force: false, opacity: 0.92,
      });
    }
    placeLabels(items, this.camera, this.canvas);
  };

  /* ------------------------------------------------------------------
     视图四：实时驾驶 3D 场景（追尾 / 车头 / 俯视 / 环绕）
     ------------------------------------------------------------------ */
  function driveCenter(harmonics, z) {
    var v = 0;
    for (var i = 0; i < harmonics.length; i++) {
      v += harmonics[i][0] * Math.sin(2 * Math.PI * z / harmonics[i][1] + (harmonics[i][2] || 0));
    }
    return v;
  }

  function driveSkyTexture(dark) {
    var cv = document.createElement('canvas');
    cv.width = 8; cv.height = 256;
    var g = cv.getContext('2d');
    var grd = g.createLinearGradient(0, 0, 0, 256);
    if (dark) {
      grd.addColorStop(0, '#0b1220'); grd.addColorStop(0.6, '#18222f'); grd.addColorStop(1, '#2b3542');
    } else {
      grd.addColorStop(0, '#bcd7f2'); grd.addColorStop(0.55, '#dcebf7'); grd.addColorStop(1, '#f2f5f7');
    }
    g.fillStyle = grd; g.fillRect(0, 0, 8, 256);
    return new THREE.CanvasTexture(cv);
  }

  function disposeChildren(group) {
    while (group.children.length) {
      var c = group.children[0];
      group.remove(c);
      c.traverse(function (o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          if (Array.isArray(o.material)) o.material.forEach(function (m) { m.dispose(); });
          else o.material.dispose();
        }
      });
    }
  }

  function makeAsphaltTexture() {
    var cv = document.createElement('canvas');
    cv.width = cv.height = 128;
    var g = cv.getContext('2d');
    g.fillStyle = '#cfd0d2';
    g.fillRect(0, 0, 128, 128);
    var rng = Math.random;
    for (var i = 0; i < 2200; i++) {
      var v = Math.floor(150 + rng() * 90);
      g.fillStyle = 'rgba(' + v + ',' + v + ',' + (v + 2) + ',' + (0.12 + rng() * 0.16).toFixed(2) + ')';
      g.fillRect(rng() * 128, rng() * 128, 1 + rng() * 2, 1 + rng() * 2);
    }
    var tex = new THREE.CanvasTexture(cv);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = 8;
    return tex;
  }

  function carShadowTexture() {
    var cv = document.createElement('canvas');
    cv.width = cv.height = 64;
    var g = cv.getContext('2d');
    var grd = g.createRadialGradient(32, 32, 2, 32, 32, 32);
    grd.addColorStop(0, 'rgba(0,0,0,0.62)');
    grd.addColorStop(0.55, 'rgba(0,0,0,0.28)');
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(cv);
  }

  var CAR_PALETTE = ['#e0564a', '#f0a030', '#4ec5d8', '#8b7cf0',
                     '#5fc46b', '#e86aa8', '#f2c14e', '#4d8ef7'];

  function DriveScene(canvas) {
    BaseScene.call(this, canvas, null);
    this.orbit.enabled = false;
    /* 驾驶场景不用实时阴影：光源跟车移动会让阴影贴图边缘持续抖动 */
    this.renderer.shadowMap.enabled = false;
    this.dir.castShadow = false;
    this.camera.fov = 58;
    this.camera.near = 0.5;
    this.camera.far = 1500;
    this.camera.updateProjectionMatrix();
    this.cameraMode = 'chase';
    this.followId = 0;
    this.data = null;
    this.poses = null;
    this.cars = {};
    this._roadKey = '';
    this._camInit = false;
    this._t = 0;
    this.camPos = new THREE.Vector3();
    this.camLook = new THREE.Vector3();
    this.roadGroup = new THREE.Group();
    this.scene.add(this.roadGroup);
    this.carGroup = new THREE.Group();
    this.scene.add(this.carGroup);

    /* 被跟随车辆的空中标记：光柱 + 向下箭头 */
    this.followMarker = new THREE.Group();
    var mBeam = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.045, 1.3, 8),
      new THREE.MeshBasicMaterial({ color: 0xf0a030, transparent: true, opacity: 0.82 })
    );
    mBeam.position.y = 2.85;
    var mTip = new THREE.Mesh(
      new THREE.ConeGeometry(0.16, 0.34, 10),
      new THREE.MeshBasicMaterial({ color: 0xf0a030 })
    );
    mTip.rotation.x = Math.PI;
    mTip.position.y = 2.10;
    var mRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.05, 0.055, 8, 32),
      new THREE.MeshBasicMaterial({ color: 0xf0a030, transparent: true, opacity: 0.85 })
    );
    mRing.rotation.x = Math.PI / 2;
    mRing.position.y = 0.15;
    this.followMarker.add(mBeam, mTip, mRing);
    this.scene.add(this.followMarker);

    /* 车模几何 / 共享材质 */
    this._shadowTex = carShadowTexture();
    this._geo = {
      body: new THREE.BoxGeometry(1.30, 0.42, 2.40),
      cabin: new THREE.BoxGeometry(1.00, 0.34, 1.10),
      wheel: new THREE.CylinderGeometry(0.22, 0.22, 0.16, 10),
      lamp: new THREE.BoxGeometry(0.22, 0.10, 0.05),
    };
    this._geo.wheel.rotateZ(Math.PI / 2);
    this._matWheel = new THREE.MeshStandardMaterial({ color: 0x111216, roughness: 0.92 });
    this._matGlass = new THREE.MeshStandardMaterial({ color: 0x1b2530, roughness: 0.18, metalness: 0.65 });
    this._matLamp = new THREE.MeshBasicMaterial({ color: 0xfff0b0 });

    /* 草地 / 大地面，接住车身阴影 */
    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(40000, 40000),
      new THREE.MeshStandardMaterial({ color: 0xdfe6df, roughness: 1.0, metalness: 0.0 })
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -0.02;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    this.onTheme();
  }
  DriveScene.prototype = Object.create(BaseScene.prototype);
  DriveScene.prototype.constructor = DriveScene;

  DriveScene.prototype.onTheme = function () {
    var dark = themeName() === 'dark';
    if (this._skyTex) { try { this._skyTex.dispose(); } catch (e) { } }
    this._skyTex = driveSkyTexture(dark);
    this.scene.background = this._skyTex;
    this.scene.fog = new THREE.Fog(dark ? 0x111a24 : 0xd7e6f2, 70, 300);
    if (this.ground) this.ground.material.color.set(dark ? 0x161d18 : 0xdfe6df);
    if (this._roadMat) this._roadMat.color.set(dark ? 0x4a4e55 : 0x6a6e75);
    if (this._treeCrownMat) this._treeCrownMat.color.set(dark ? 0x2b5a36 : 0x3f7d4b);
    if (this._treeTrunkMat) this._treeTrunkMat.color.set(dark ? 0x4a3a29 : 0x6b4f35);
    if (this._lastRoad && this.roadGroup && this.roadGroup.children.length) {
      this._roadKey = '';
      this.setRoad(this._lastRoad);
    }
  };

  DriveScene.prototype.setCameraMode = function (mode) { this.cameraMode = mode || 'chase'; };
  DriveScene.prototype.setFollow = function (id) { this.followId = id; };
  DriveScene.prototype.resetCamera = function () { this._camInit = false; };

  DriveScene.prototype._makeCar = function (colorHex) {
    var self = this;
    var g = new THREE.Group();
    /* 稳定软阴影贴片（不依赖阴影贴图，不会随光源移动闪烁） */
    var blob = new THREE.Mesh(
      new THREE.PlaneGeometry(2.0, 3.0),
      new THREE.MeshBasicMaterial({
        map: this._shadowTex, transparent: true, opacity: 0.55, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
      })
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.015;
    g.add(blob);

    var bodyMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(colorHex), roughness: 0.32, metalness: 0.42,
    });
    var body = new THREE.Mesh(this._geo.body, bodyMat);
    body.position.y = 0.46; body.castShadow = true; g.add(body);

    var cabin = new THREE.Mesh(this._geo.cabin, this._matGlass);
    cabin.position.set(0, 0.80, 0.10); cabin.castShadow = true; g.add(cabin);

    var wx = 0.66, wz = 0.80;
    [[-wx, wz], [wx, wz], [-wx, -wz], [wx, -wz]].forEach(function (p) {
      var w = new THREE.Mesh(self._geo.wheel, self._matWheel);
      w.position.set(p[0], 0.22, p[1]); w.castShadow = true; g.add(w);
    });

    [-0.38, 0.38].forEach(function (x) {
      var lamp = new THREE.Mesh(self._geo.lamp, self._matLamp);
      lamp.position.set(x, 0.50, 1.21); g.add(lamp);
    });
    return g;
  };

  DriveScene.prototype.setRoad = function (road) {
    if (!road) return;
    var key = JSON.stringify(road.harmonics || []);
    if (key === this._roadKey && this.roadGroup.children.length) return;
    this._roadKey = key;
    this._lastRoad = road;
    disposeChildren(this.roadGroup);

    var harmonics = road.harmonics || [[12, 300, 0]];
    var half = road.half_w || 10;
    var dark = themeName() === 'dark';
    var Z0 = -80, Z1 = 12000, STEP = 6;

    /* 路面：沿 z 的带状网格 */
    var verts = [], uvs = [], idx = [], n = 0;
    for (var z = Z0; z <= Z1; z += STEP) {
      var c = driveCenter(harmonics, z);
      verts.push(c - half, 0.05, z, c + half, 0.05, z);
      uvs.push(0, z / 20, 1, z / 20);
      if (n > 0) {
        var a = (n - 1) * 2, b = n * 2;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
      n++;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    if (this._roadTex) { try { this._roadTex.dispose(); } catch (e) { } }
    this._roadTex = makeAsphaltTexture();
    this._roadMat = new THREE.MeshStandardMaterial({
      color: dark ? 0x4a4e55 : 0x6a6e75, roughness: 0.96, metalness: 0.02,
      map: this._roadTex,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    var roadMesh = new THREE.Mesh(geo, this._roadMat);
    roadMesh.receiveShadow = true;
    this.roadGroup.add(roadMesh);

    /* 两侧边线 + 中心虚线 */
    var leftPts = [], rightPts = [], midPts = [];
    for (var z2 = Z0; z2 <= Z1; z2 += 8) {
      var c2 = driveCenter(harmonics, z2);
      leftPts.push(new THREE.Vector3(c2 - half + 0.45, 0.12, z2));
      rightPts.push(new THREE.Vector3(c2 + half - 0.45, 0.12, z2));
      midPts.push(new THREE.Vector3(c2, 0.12, z2));
    }
    var edgeMat = new THREE.LineBasicMaterial({
      color: dark ? 0xc9d2dc : 0xf2f2f2, transparent: true, opacity: 0.9,
      polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
    });
    var edgeL = new THREE.Line(new THREE.BufferGeometry().setFromPoints(leftPts), edgeMat);
    var edgeR = new THREE.Line(new THREE.BufferGeometry().setFromPoints(rightPts), edgeMat);
    this.roadGroup.add(edgeL, edgeR);
    var midGeo = new THREE.BufferGeometry().setFromPoints(midPts);
    var midLine = new THREE.Line(midGeo, new THREE.LineDashedMaterial({
      color: 0xe8c45a, dashSize: 4, gapSize: 4, transparent: true, opacity: 0.95,
      polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
    }));
    midLine.computeLineDistances();
    this.roadGroup.add(midLine);

    /* 路边树木：InstancedMesh，一次绘制大量树 */
    if (THREE.InstancedMesh) {
      var count = 220;
      var trunkGeo = new THREE.CylinderGeometry(0.13, 0.18, 1.5, 6);
      trunkGeo.translate(0, 0.75, 0);
      var crownGeo = new THREE.ConeGeometry(0.95, 2.6, 7);
      crownGeo.translate(0, 2.0, 0);
      this._treeTrunkMat = new THREE.MeshStandardMaterial({
        color: dark ? 0x4a3a29 : 0x6b4f35, roughness: 1.0,
      });
      this._treeCrownMat = new THREE.MeshStandardMaterial({
        color: dark ? 0x2b5a36 : 0x3f7d4b, roughness: 0.95,
      });
      var trunks = new THREE.InstancedMesh(trunkGeo, this._treeTrunkMat, count);
      var crowns = new THREE.InstancedMesh(crownGeo, this._treeCrownMat, count);
      /* 树木不投影，避免光源跟车移动时阴影边缘闪烁 */
      trunks.castShadow = false; crowns.castShadow = false;
      var dummy = new THREE.Object3D();
      for (var i = 0; i < count; i++) {
        var tz = -60 + i * 52;
        var side = (i % 2) ? 1 : -1;
        var off = half + 7.5 + (i * 37 % 8);
        var tx = driveCenter(harmonics, tz) + side * off;
        var s = 0.8 + (i * 13 % 7) * 0.08;
        dummy.position.set(tx, 0, tz);
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        trunks.setMatrixAt(i, dummy.matrix);
        crowns.setMatrixAt(i, dummy.matrix);
      }
      trunks.instanceMatrix.needsUpdate = true;
      crowns.instanceMatrix.needsUpdate = true;
      this.roadGroup.add(trunks, crowns);
    }
  };

  DriveScene.prototype.setState = function (data, poses) {
    if (!data) return;
    this.data = data;
    this.poses = poses || {};
    if (data.road) this.setRoad(data.road);
    if (!data.cars) return;
    var self = this;
    var seen = {};
    data.cars.forEach(function (c, i) {
      seen[c.id] = true;
      var car = self.cars[c.id];
      if (!car) {
        car = self._makeCar(CAR_PALETTE[i % CAR_PALETTE.length]);
        self.cars[c.id] = car;
        self.carGroup.add(car);
      }
      var p = (poses && poses[c.id]) || c;
      car.position.set(p.x, 0.05, p.z);
      car.rotation.y = p.theta || 0;
      car.visible = true;
    });
    for (var id in this.cars) {
      if (!seen[id]) this.cars[id].visible = false;
    }
  };

  DriveScene.prototype._updateCamera = function (dt) {
    if (!this.data || !this.data.cars || !this.data.cars.length) return;
    var cars = this.data.cars;
    var car = cars[0];
    for (var i = 0; i < cars.length; i++) if (cars[i].id === this.followId) car = cars[i];
    var p = (this.poses && this.poses[car.id]) || car;
    var th = p.theta || 0;
    var fx = Math.sin(th), fz = Math.cos(th);
    var pos = new THREE.Vector3(), look = new THREE.Vector3();

    if (this.cameraMode === 'hood') {
      pos.set(p.x + fx * 0.20, 1.18, p.z + fz * 0.20);
      look.set(p.x + fx * 16, 0.75, p.z + fz * 16);
    } else if (this.cameraMode === 'top') {
      pos.set(p.x, 26, p.z - 5);
      look.set(p.x + fx * 6, 0, p.z + fz * 6 + 4);
    } else if (this.cameraMode === 'orbit') {
      var a = this._t * 0.42;
      pos.set(p.x + Math.cos(a) * 9.5, 5.4, p.z + Math.sin(a) * 9.5);
      look.set(p.x, 0.65, p.z);
    } else {
      pos.set(p.x - fx * 5.2, 2.8, p.z - fz * 5.2);
      look.set(p.x + fx * 11, 0.80, p.z + fz * 11);
    }

    if (this.cameraMode === 'orbit') {
      var k = this._camInit ? (1 - Math.exp(-dt * 6)) : 1;
      this.camPos.lerp(pos, k);
      this.camLook.lerp(look, k);
    } else {
      /* 追尾/车头/俯视：刚性跟随已平滑的位姿，避免二次缓动造成周期性抽动 */
      this.camPos.copy(pos);
      this.camLook.copy(look);
    }
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
    this._camInit = true;

    /* 标记当前跟随的车辆 */
    if (this.followMarker) {
      this.followMarker.position.set(p.x, 0, p.z);
    }

    /* 阴影光源跟随当前车辆，保证远处也有阴影 */
    this.dir.target.position.set(p.x, 0, p.z);
    this.dir.position.set(p.x + 24, 42, p.z + 18);
    this.dir.target.updateMatrixWorld();
  };

  DriveScene.prototype.update = function (dt) {
    this._t += dt;
    this._updateCamera(dt);
  };


  var SCENES = { arch: ArchScene, weights: WeightScene, backward: BackwardScene };

  function create(mode, canvas, overlay) {
    var S = SCENES[mode] || SCENES.arch;
    return new S(canvas, overlay);
  }

  global.MF3D = {
    create: create,
    createDrive: function (canvas) { return new DriveScene(canvas); },
    modes: ['arch', 'weights', 'backward'],
    applyTheme: function (scene) { if (scene && scene.applyTheme) scene.applyTheme(); },
  };
})(window);
