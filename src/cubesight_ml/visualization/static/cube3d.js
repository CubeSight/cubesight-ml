/**
 * CubeSight-ML 3D Rubik's Cube Renderer with Three.js
 */

const PALETTE = {
  0: 0xffffff, // White (U)
  1: 0xdc2626, // Red (R)
  2: 0x16a34a, // Green (F)
  3: 0xfacc15, // Yellow (D)
  4: 0xea580c, // Orange (L)
  5: 0x2563eb, // Blue (B)
};

const PLASTIC_BLACK = 0x18181b;

class RubiksCube3D {
  constructor(containerElement) {
    this.container = containerElement;
    this.cubies = []; // 27 mesh objects
    this.cubieMap = new Map();
    this.isAnimating = false;
    this.animFrameId = null;

    this.initScene();
    this.initCubies();
    this.setupResize();
    this.animate();
  }

  initScene() {
    this.scene = new THREE.Scene();

    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    this.camera.position.set(4.5, 4.0, 5.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 3.5;
    this.controls.maxDistance = 12.0;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight1.position.set(6, 10, 8);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.35);
    dirLight2.position.set(-6, -8, -6);
    this.scene.add(dirLight2);

    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);
  }

  initCubies() {
    const geom = new THREE.BoxGeometry(0.96, 0.96, 0.96);
    this.cubies = [];
    this.cubieMap = new Map();

    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
          const materials = [
            this.createFaceMaterial(PLASTIC_BLACK), // +X (Right)
            this.createFaceMaterial(PLASTIC_BLACK), // -X (Left)
            this.createFaceMaterial(PLASTIC_BLACK), // +Y (Top)
            this.createFaceMaterial(PLASTIC_BLACK), // -Y (Bottom)
            this.createFaceMaterial(PLASTIC_BLACK), // +Z (Front)
            this.createFaceMaterial(PLASTIC_BLACK), // -Z (Back)
          ];

          const mesh = new THREE.Mesh(geom, materials);
          mesh.position.set(x, y, z);
          mesh.castShadow = true;
          mesh.receiveShadow = true;

          mesh.userData = {
            gridX: x,
            gridY: y,
            gridZ: z,
          };

          this.cubies.push(mesh);
          this.cubieMap.set(`${x},${y},${z}`, mesh);
          this.scene.add(mesh);
        }
      }
    }
  }

  createFaceMaterial(colorHex) {
    return new THREE.MeshStandardMaterial({
      color: colorHex,
      roughness: 0.25,
      metalness: 0.05,
    });
  }

  getFaceColor(colorId) {
    if (colorId !== null && colorId !== undefined && PALETTE[colorId] !== undefined) {
      return PALETTE[colorId];
    }
    return PLASTIC_BLACK;
  }

  resetCubieTransforms() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    this.isAnimating = false;

    // 1. Return any cubies attached to pivot back to scene
    while (this.pivot.children.length > 0) {
      this.scene.add(this.pivot.children[0]);
    }

    // 2. Reset pivot transform to identity
    this.pivot.position.set(0, 0, 0);
    this.pivot.rotation.set(0, 0, 0);
    this.pivot.quaternion.identity();
    this.pivot.scale.set(1, 1, 1);
    this.pivot.updateMatrix();
    this.pivot.updateMatrixWorld(true);

    // 3. Reset every cubie mesh to canonical grid position and identity rotation
    for (const c of this.cubies) {
      c.position.set(c.userData.gridX, c.userData.gridY, c.userData.gridZ);
      c.rotation.set(0, 0, 0);
      c.quaternion.identity();
      c.scale.set(1, 1, 1);
      c.updateMatrix();
      c.updateMatrixWorld(true);
    }
  }

  updateColors(cubiesData) {
    if (!cubiesData || cubiesData.length !== 27) return;

    // Guarantee local axes are strictly aligned with world axes
    this.resetCubieTransforms();

    for (const data of cubiesData) {
      const cubie = this.cubieMap.get(`${data.x},${data.y},${data.z}`);
      if (!cubie) continue;

      const faces = data.faces || {};
      // Box face order: Right(+X), Left(-X), Top(+Y), Bottom(-Y), Front(+Z), Back(-Z)
      const colors = [
        this.getFaceColor(faces.right),
        this.getFaceColor(faces.left),
        this.getFaceColor(faces.top),
        this.getFaceColor(faces.bottom),
        this.getFaceColor(faces.front),
        this.getFaceColor(faces.back),
      ];

      for (let i = 0; i < 6; i++) {
        cubie.material[i].color.setHex(colors[i]);
      }
    }
  }

  getMoveParameters(move) {
    // Returns { axis: 'x'|'y'|'z', filter: (cubie) => bool, angle: radians }
    // Quarter turn = Math.PI / 2
    const Q = Math.PI / 2;

    const baseMove = move.replace(/[2']/g, "");
    const isDouble = move.includes("2");
    const isPrime = move.includes("'");
    let sign = isPrime ? 1 : -1;
    if (isDouble) sign = -2;

    switch (baseMove) {
      // Face turns
      case "U":
        return { axis: "y", angle: sign * Q, filter: (p) => p.y > 0.5 };
      case "D":
        return { axis: "y", angle: -sign * Q, filter: (p) => p.y < -0.5 };
      case "R":
        return { axis: "x", angle: sign * Q, filter: (p) => p.x > 0.5 };
      case "L":
        return { axis: "x", angle: -sign * Q, filter: (p) => p.x < -0.5 };
      case "F":
        return { axis: "z", angle: sign * Q, filter: (p) => p.z > 0.5 };
      case "B":
        return { axis: "z", angle: -sign * Q, filter: (p) => p.z < -0.5 };

      // Slices (M follows L, E follows D, S follows F)
      case "M":
        return { axis: "x", angle: -sign * Q, filter: (p) => Math.abs(p.x) < 0.5 };
      case "E":
        return { axis: "y", angle: -sign * Q, filter: (p) => Math.abs(p.y) < 0.5 };
      case "S":
        return { axis: "z", angle: sign * Q, filter: (p) => Math.abs(p.z) < 0.5 };

      // Wide turns
      case "r":
        return { axis: "x", angle: sign * Q, filter: (p) => p.x > -0.5 };
      case "l":
        return { axis: "x", angle: -sign * Q, filter: (p) => p.x < 0.5 };
      case "u":
        return { axis: "y", angle: sign * Q, filter: (p) => p.y > -0.5 };
      case "d":
        return { axis: "y", angle: -sign * Q, filter: (p) => p.y < 0.5 };
      case "f":
        return { axis: "z", angle: sign * Q, filter: (p) => p.z > -0.5 };
      case "b":
        return { axis: "z", angle: -sign * Q, filter: (p) => p.z < 0.5 };

      // Whole cube rotations
      case "x":
        return { axis: "x", angle: sign * Q, filter: () => true };
      case "y":
        return { axis: "y", angle: sign * Q, filter: () => true };
      case "z":
        return { axis: "z", angle: sign * Q, filter: () => true };

      default:
        return null;
    }
  }

  animateMove(move, durationMs = 300, onFinished = null) {
    if (this.isAnimating) {
      this.resetCubieTransforms();
    }

    const params = this.getMoveParameters(move);
    if (!params) {
      if (onFinished) onFinished();
      return;
    }

    this.isAnimating = true;

    // 1. Group cubies satisfying filter to pivot
    this.pivot.position.set(0, 0, 0);
    this.pivot.rotation.set(0, 0, 0);
    this.pivot.quaternion.identity();
    this.pivot.updateMatrixWorld(true);

    const rotatingCubies = this.cubies.filter((c) => params.filter(c.position));
    for (const c of rotatingCubies) {
      this.pivot.attach(c);
    }

    // 2. Animate rotation
    const startTime = performance.now();
    const targetAngle = params.angle;

    const stepAnimation = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / Math.max(durationMs, 1), 1.0);

      // Smooth ease-out quad
      const ease = 1 - (1 - progress) * (1 - progress);
      const currentAngle = targetAngle * ease;

      if (params.axis === "x") this.pivot.rotation.x = currentAngle;
      if (params.axis === "y") this.pivot.rotation.y = currentAngle;
      if (params.axis === "z") this.pivot.rotation.z = currentAngle;

      if (progress < 1.0) {
        this.animFrameId = requestAnimationFrame(stepAnimation);
      } else {
        // Complete rotation
        if (params.axis === "x") this.pivot.rotation.x = targetAngle;
        if (params.axis === "y") this.pivot.rotation.y = targetAngle;
        if (params.axis === "z") this.pivot.rotation.z = targetAngle;

        this.pivot.updateMatrixWorld(true);
        this.animFrameId = null;
        this.isAnimating = false;

        if (onFinished) {
          try {
            onFinished();
          } catch (err) {
            console.error("Error in onFinished callback:", err);
            this.resetCubieTransforms();
          }
        } else {
          this.resetCubieTransforms();
        }
      }
    };

    this.animFrameId = requestAnimationFrame(stepAnimation);
  }

  setupResize() {
    window.addEventListener("resize", () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
