/* ============================================================
   Spiderweb Visualization — Canvas-based Agent Network
   Shows the QuantumSpiderweb as an animated node graph.
   Zero dependencies. Pure Canvas API.
   ============================================================ */

class SpiderwebViz {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = {};
        this.attractors = [];
        this.coherence = 0;
        this.animFrame = null;
        this.time = 0;

        // Agent positions (circular layout)
        this.agents = [
            'newton', 'davinci', 'empathy', 'philosophy',
            'quantum', 'consciousness', 'multi_perspective', 'systems_architecture'
        ];

        this.colors = {
            newton: '#3b82f6', davinci: '#f59e0b', empathy: '#a855f7',
            philosophy: '#10b981', quantum: '#ef4444', consciousness: '#e2e8f0',
            multi_perspective: '#f97316', systems_architecture: '#06b6d4',
        };

        this.labels = {
            newton: 'N', davinci: 'D', empathy: 'E', philosophy: 'P',
            quantum: 'Q', consciousness: 'C', multi_perspective: 'M',
            systems_architecture: 'S',
        };

        // Initialize with default state
        this._initDefaultState();
        this._resize();
        this._animate();

        // Handle resize
        new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    }

    _initDefaultState() {
        this.agents.forEach((name, i) => {
            this.nodes[name] = {
                state: [0.5, 0, 0.5, 0, 0.5],  // psi, tau, chi, phi, lam
                tension: 0,
                active: false,
                energy: 0.25,
            };
        });
    }

    _resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = 200 * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = '200px';
        this.ctx.scale(dpr, dpr);
        this.w = rect.width;
        this.h = 200;
        this.cx = this.w / 2;
        this.cy = this.h / 2;
        this.radius = Math.min(this.w, this.h) * 0.35;
    }

    update(spiderwebState) {
        if (!spiderwebState || !spiderwebState.nodes) return;

        // Update node states
        for (const [name, data] of Object.entries(spiderwebState.nodes)) {
            if (this.nodes[name]) {
                this.nodes[name].state = data.state || [0.5, 0, 0.5, 0, 0.5];
                const tensions = data.tension_history || [];
                this.nodes[name].tension = tensions.length > 0 ?
                    tensions[tensions.length - 1] : 0;
                this.nodes[name].energy = data.state ?
                    data.state.reduce((s, v) => s + v * v, 0) : 0.25;
                this.nodes[name].active = (data.state[0] || 0) > 0.6;
            }
        }

        this.attractors = spiderwebState.attractors || [];
        this.coherence = spiderwebState.phase_coherence || 0;
    }

    _getNodePos(index) {
        const angle = (index / this.agents.length) * Math.PI * 2 - Math.PI / 2;
        return {
            x: this.cx + Math.cos(angle) * this.radius,
            y: this.cy + Math.sin(angle) * this.radius,
        };
    }

    _animate() {
        this.time += 0.016;
        this._draw();
        this.animFrame = requestAnimationFrame(() => this._animate());
    }

    _draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.w, this.h);

        // Background glow based on coherence
        if (this.coherence > 0.5) {
            const gradient = ctx.createRadialGradient(
                this.cx, this.cy, 0, this.cx, this.cy, this.radius * 1.5
            );
            gradient.addColorStop(0, `rgba(59, 130, 246, ${this.coherence * 0.05})`);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, this.w, this.h);
        }

        // Draw edges (connections between all agents)
        this.agents.forEach((nameA, i) => {
            const posA = this._getNodePos(i);
            this.agents.forEach((nameB, j) => {
                if (j <= i) return;
                const posB = this._getNodePos(j);

                // Tension between nodes determines edge style
                const nodeA = this.nodes[nameA];
                const nodeB = this.nodes[nameB];
                const tension = Math.abs((nodeA?.tension || 0) - (nodeB?.tension || 0));

                ctx.beginPath();
                ctx.moveTo(posA.x, posA.y);
                ctx.lineTo(posB.x, posB.y);

                // Higher tension = more visible edge
                const alpha = 0.05 + Math.min(tension * 0.3, 0.2);

                // Pulse active connections
                const bothActive = nodeA?.active && nodeB?.active;
                const pulseAlpha = bothActive ?
                    alpha + Math.sin(this.time * 3) * 0.1 : alpha;

                ctx.strokeStyle = bothActive ?
                    `rgba(168, 85, 247, ${pulseAlpha})` :
                    `rgba(148, 163, 184, ${pulseAlpha})`;
                ctx.lineWidth = bothActive ? 1.5 : 0.5;
                ctx.stroke();
            });
        });

        // Draw attractor regions
        this.attractors.forEach((att, ai) => {
            if (!att.members || att.members.length < 2) return;

            // Find centroid of member nodes
            let cx = 0, cy = 0, count = 0;
            att.members.forEach(name => {
                const idx = this.agents.indexOf(name);
                if (idx >= 0) {
                    const pos = this._getNodePos(idx);
                    cx += pos.x;
                    cy += pos.y;
                    count++;
                }
            });
            if (count < 2) return;
            cx /= count;
            cy /= count;

            // Draw attractor cloud
            const attRadius = 20 + count * 8;
            const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, attRadius);
            gradient.addColorStop(0, `rgba(168, 85, 247, ${0.08 + Math.sin(this.time * 2 + ai) * 0.03})`);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(cx, cy, attRadius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw nodes
        this.agents.forEach((name, i) => {
            const pos = this._getNodePos(i);
            const node = this.nodes[name];
            const color = this.colors[name] || '#94a3b8';
            const energy = node?.energy || 0.25;
            const isActive = node?.active || false;

            // Node glow
            if (isActive) {
                const glowRadius = 12 + Math.sin(this.time * 2) * 3;
                const glow = ctx.createRadialGradient(
                    pos.x, pos.y, 0, pos.x, pos.y, glowRadius
                );
                glow.addColorStop(0, color + '40');
                glow.addColorStop(1, 'transparent');
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
                ctx.fill();
            }

            // Node circle
            const nodeRadius = 6 + energy * 4;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = isActive ? color : color + '60';
            ctx.fill();

            if (isActive) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Label
            ctx.fillStyle = isActive ? '#e2e8f0' : '#64748b';
            ctx.font = `${isActive ? 'bold ' : ''}9px system-ui`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(this.labels[name], pos.x, pos.y + nodeRadius + 10);
        });

        // Coherence ring
        if (this.coherence > 0) {
            ctx.beginPath();
            ctx.arc(this.cx, this.cy, this.radius + 15, 0, Math.PI * 2 * this.coherence);
            ctx.strokeStyle = `rgba(16, 185, 129, ${0.2 + this.coherence * 0.3})`;
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Coherence label
            ctx.fillStyle = '#64748b';
            ctx.font = '9px system-ui';
            ctx.textAlign = 'center';
            ctx.fillText(`\u0393 ${this.coherence.toFixed(2)}`, this.cx, this.h - 8);
        }
    }

    destroy() {
        if (this.animFrame) cancelAnimationFrame(this.animFrame);
    }
}
