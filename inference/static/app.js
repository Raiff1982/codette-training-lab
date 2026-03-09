/* ============================================================
   Codette Chat UI — Frontend Logic
   Pure vanilla JS. Zero dependencies.
   ============================================================ */

// Adapter color map
const COLORS = {
    newton: '#3b82f6', davinci: '#f59e0b', empathy: '#a855f7',
    philosophy: '#10b981', quantum: '#ef4444', consciousness: '#e2e8f0',
    multi_perspective: '#f97316', systems_architecture: '#06b6d4',
    _base: '#94a3b8', auto: '#94a3b8',
};

const LABELS = {
    newton: 'N', davinci: 'D', empathy: 'E', philosophy: 'P',
    quantum: 'Q', consciousness: 'C', multi_perspective: 'M',
    systems_architecture: 'S',
};

// State
let isLoading = false;
let spiderwebViz = null;

// ── Initialization ──
document.addEventListener('DOMContentLoaded', () => {
    initUI();
    pollStatus();
    loadSessions();
    initCoverageDots();
    initAdapterDots();

    // Initialize spiderweb canvas
    const canvas = document.getElementById('spiderweb-canvas');
    if (canvas) {
        spiderwebViz = new SpiderwebViz(canvas);
    }
});

function initUI() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const newBtn = document.getElementById('btn-new-chat');
    const panelBtn = document.getElementById('btn-toggle-panel');
    const maxAdapters = document.getElementById('max-adapters');

    // Send on Enter (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    sendBtn.addEventListener('click', sendMessage);
    newBtn.addEventListener('click', newChat);

    panelBtn.addEventListener('click', () => {
        const panel = document.getElementById('side-panel');
        panel.classList.toggle('collapsed');
        // Update button label
        panelBtn.textContent = panel.classList.contains('collapsed') ? 'Cocoon' : 'Close';
    });

    maxAdapters.addEventListener('input', () => {
        document.getElementById('max-adapters-value').textContent = maxAdapters.value;
    });

    // Voice input via Web Speech API
    initVoice(micBtn);

    // TTS toggle — read responses aloud when enabled
    const ttsToggle = document.getElementById('tts-toggle');
    if (ttsToggle) {
        ttsToggle.addEventListener('change', () => {
            if (ttsToggle.checked && !window.speechSynthesis) {
                ttsToggle.checked = false;
                ttsToggle.parentElement.title = 'Speech synthesis not supported';
            }
        });
    }
}

// ── Voice Input ──
let _recognition = null;
let _isRecording = false;

function initVoice(micBtn) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        micBtn.title = 'Voice not supported in this browser';
        micBtn.style.opacity = '0.3';
        micBtn.style.cursor = 'not-allowed';
        return;
    }

    _recognition = new SpeechRecognition();
    _recognition.continuous = false;
    _recognition.interimResults = true;
    _recognition.lang = 'en-US';

    const input = document.getElementById('chat-input');

    _recognition.onstart = () => {
        _isRecording = true;
        micBtn.classList.add('recording');
        micBtn.title = 'Listening... click to stop';
    };

    _recognition.onresult = (event) => {
        let transcript = '';
        let isFinal = false;
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
            if (event.results[i].isFinal) isFinal = true;
        }
        // Show interim results in the input box
        input.value = transcript;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';

        if (isFinal) {
            stopVoice(micBtn);
        }
    };

    _recognition.onerror = (event) => {
        console.log('Speech recognition error:', event.error);
        stopVoice(micBtn);
        if (event.error === 'not-allowed') {
            micBtn.title = 'Microphone access denied';
        }
    };

    _recognition.onend = () => {
        stopVoice(micBtn);
    };

    micBtn.addEventListener('click', () => {
        if (_isRecording) {
            _recognition.stop();
            stopVoice(micBtn);
        } else {
            try {
                _recognition.start();
            } catch (e) {
                console.log('Speech recognition start error:', e);
            }
        }
    });
}

function stopVoice(micBtn) {
    _isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.title = 'Voice input';
}

// ── Status Polling ──
function pollStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(status => {
            updateStatus(status);
            if (status.state === 'loading') {
                setTimeout(pollStatus, 2000);
            } else if (status.state === 'ready') {
                hideLoadingScreen();
            } else if (status.state === 'error') {
                // Model failed to load — show error and dismiss loading screen
                hideLoadingScreen();
                updateStatus({ state: 'error', message: status.message || 'Model failed to load' });
            } else if (status.state === 'idle') {
                // Model not loaded yet, keep polling
                setTimeout(pollStatus, 3000);
            }
        })
        .catch(() => {
            setTimeout(pollStatus, 5000);
        });
}

function updateStatus(status) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');

    dot.className = 'status-dot ' + (status.state || 'loading');
    text.textContent = status.message || status.state;

    // Update loading screen
    const loadingStatus = document.getElementById('loading-status');
    if (loadingStatus) {
        loadingStatus.textContent = status.message || 'Loading...';
    }

    // Update adapter dots if available
    if (status.adapters) {
        updateAdapterDots(status.adapters);
    }
}

function hideLoadingScreen() {
    const screen = document.getElementById('loading-screen');
    if (screen) {
        screen.classList.add('hidden');
        setTimeout(() => screen.remove(), 500);
    }
}

// ── Adapter Dots ──
function initAdapterDots() {
    const container = document.getElementById('adapter-dots');
    Object.keys(LABELS).forEach(name => {
        const dot = document.createElement('span');
        dot.className = 'adapter-dot';
        dot.style.backgroundColor = COLORS[name];
        dot.title = name;
        dot.id = `dot-${name}`;
        container.appendChild(dot);
    });
}

function updateAdapterDots(available) {
    Object.keys(LABELS).forEach(name => {
        const dot = document.getElementById(`dot-${name}`);
        if (dot) {
            dot.classList.toggle('available', available.includes(name));
        }
    });
}

function setActiveAdapter(name) {
    // Remove previous active
    document.querySelectorAll('.adapter-dot').forEach(d => d.classList.remove('active'));
    // Set new active
    const dot = document.getElementById(`dot-${name}`);
    if (dot) dot.classList.add('active');

    // Update CSS accent color
    const color = COLORS[name] || COLORS._base;
    document.documentElement.style.setProperty('--accent', color);
    document.documentElement.style.setProperty('--accent-glow', color + '25');
}

// ── Coverage Dots ──
function initCoverageDots() {
    const container = document.getElementById('coverage-dots');
    Object.entries(LABELS).forEach(([name, label]) => {
        const dot = document.createElement('span');
        dot.className = 'coverage-dot';
        dot.style.color = COLORS[name];
        dot.textContent = label;
        dot.title = name;
        dot.id = `cov-${name}`;
        container.appendChild(dot);
    });
}

function updateCoverage(usage) {
    Object.keys(LABELS).forEach(name => {
        const dot = document.getElementById(`cov-${name}`);
        if (dot) {
            dot.classList.toggle('active', (usage[name] || 0) > 0);
        }
    });
}

// ── Chat ──
function sendMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query || isLoading) return;

    // Hide welcome
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.style.display = 'none';

    // Add user message
    addMessage('user', query);

    // Clear input
    input.value = '';
    input.style.height = 'auto';

    // Get settings
    const adapter = document.getElementById('adapter-select').value;
    const maxAdapters = parseInt(document.getElementById('max-adapters').value);

    // Show thinking
    const thinkingEl = showThinking(adapter);
    isLoading = true;
    document.getElementById('send-btn').disabled = true;

    // Send request
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            adapter: adapter === 'auto' ? null : adapter,
            max_adapters: maxAdapters,
        }),
    })
    .then(r => r.json())
    .then(data => {
        thinkingEl.remove();

        if (data.error) {
            addMessage('error', data.error);
            return;
        }

        // Add assistant message
        const adapterUsed = data.adapter || '_base';
        setActiveAdapter(adapterUsed);

        addMessage('assistant', data.response, {
            adapter: adapterUsed,
            confidence: data.confidence,
            reasoning: data.reasoning,
            tokens: data.tokens,
            time: data.time,
            perspectives: data.perspectives,
            multi_perspective: data.multi_perspective,
            tools_used: data.tools_used,
        });

        // Speak response if TTS is enabled
        const ttsOn = document.getElementById('tts-toggle');
        if (ttsOn && ttsOn.checked && window.speechSynthesis) {
            const utter = new SpeechSynthesisUtterance(data.response);
            utter.rate = 1.0;
            utter.pitch = 1.0;
            window.speechSynthesis.speak(utter);
        }

        // Update cocoon state
        if (data.cocoon) {
            updateCocoonUI(data.cocoon);
        }

        // Update epistemic metrics
        if (data.epistemic) {
            updateEpistemicUI(data.epistemic);
        }
    })
    .catch(err => {
        thinkingEl.remove();
        addMessage('error', `Request failed: ${err.message}`);
    })
    .finally(() => {
        isLoading = false;
        document.getElementById('send-btn').disabled = false;
        document.getElementById('chat-input').focus();
    });
}

function askQuestion(query) {
    document.getElementById('chat-input').value = query;
    sendMessage();
}

function addMessage(role, content, meta = {}) {
    const area = document.getElementById('chat-area');
    const msg = document.createElement('div');
    msg.className = `message message-${role}`;

    if (role === 'user') {
        msg.innerHTML = `<div class="bubble"><div class="message-text">${escapeHtml(content)}</div></div>`;
    } else if (role === 'assistant') {
        const adapter = meta.adapter || '_base';
        const color = COLORS[adapter] || COLORS._base;
        const conf = meta.confidence || 0;
        const tps = meta.tokens && meta.time ? (meta.tokens / meta.time).toFixed(1) : '?';

        let html = `<div class="bubble" style="border-left-color:${color}">`;
        html += `<div class="message-header">`;
        html += `<span class="adapter-badge" style="color:${color}">${adapter}</span>`;
        html += `<div class="confidence-bar"><div class="confidence-fill" style="width:${conf*100}%;background:${color}"></div></div>`;
        html += `<span>${(conf*100).toFixed(0)}%</span>`;
        html += `</div>`;
        html += `<div class="message-text">${renderMarkdown(content)}</div>`;
        html += `<div class="message-meta">${meta.tokens || '?'} tokens | ${tps} tok/s | ${(meta.time||0).toFixed(1)}s</div>`;

        // Tool usage indicator
        if (meta.tools_used && meta.tools_used.length > 0) {
            const toolNames = meta.tools_used.map(t => t.tool).join(', ');
            html += `<div class="tools-badge">🔧 Tools: ${toolNames}</div>`;
        }

        // Multi-perspective expandable
        if (meta.perspectives && Object.keys(meta.perspectives).length > 1) {
            const perspId = 'persp-' + Date.now();
            html += `<button class="perspectives-toggle" onclick="togglePerspectives('${perspId}')">`;
            html += `Show ${Object.keys(meta.perspectives).length} perspectives</button>`;
            html += `<div class="perspectives-panel" id="${perspId}">`;
            for (const [name, text] of Object.entries(meta.perspectives)) {
                const pc = COLORS[name] || COLORS._base;
                html += `<div class="perspective-card" style="border-left-color:${pc}">`;
                html += `<div class="perspective-card-header" style="color:${pc}">${name}</div>`;
                html += `<div>${renderMarkdown(text)}</div></div>`;
            }
            html += `</div>`;
        }

        html += `</div>`;
        msg.innerHTML = html;
    } else if (role === 'error') {
        msg.innerHTML = `<div class="bubble" style="border-left-color:var(--quantum)">
            <div class="message-text" style="color:var(--quantum)">${escapeHtml(content)}</div></div>`;
    }

    area.appendChild(msg);
    area.scrollTop = area.scrollHeight;
}

function showThinking(adapter) {
    const area = document.getElementById('chat-area');
    const el = document.createElement('div');
    el.className = 'thinking';
    el.innerHTML = `
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <span>Codette is thinking${adapter && adapter !== 'auto' ? ` (${adapter})` : ''}...</span>
    `;
    area.appendChild(el);
    area.scrollTop = area.scrollHeight;
    return el;
}

function togglePerspectives(id) {
    document.getElementById(id).classList.toggle('open');
}

// ── Cocoon UI Updates ──
function updateCocoonUI(state) {
    // Metrics
    const metrics = state.metrics || {};
    const coherence = metrics.current_coherence || 0;
    const tension = metrics.current_tension || 0;

    document.getElementById('metric-coherence').textContent = coherence.toFixed(4);
    document.getElementById('bar-coherence').style.width = (coherence * 100) + '%';

    document.getElementById('metric-tension').textContent = tension.toFixed(4);
    document.getElementById('bar-tension').style.width = Math.min(tension * 100, 100) + '%';

    document.getElementById('cocoon-attractors').textContent = metrics.attractor_count || 0;
    document.getElementById('cocoon-glyphs').textContent = metrics.glyph_count || 0;

    // Cocoon status
    const cocoon = state.cocoon || {};
    document.getElementById('cocoon-encryption').textContent =
        cocoon.has_sync ? 'Active' : 'Available';

    // Coverage
    updateCoverage(state.perspective_usage || {});

    // Spiderweb
    if (spiderwebViz && state.spiderweb) {
        spiderwebViz.update(state.spiderweb);
    }
}

function updateEpistemicUI(epistemic) {
    if (epistemic.ensemble_coherence !== undefined) {
        const val = epistemic.ensemble_coherence;
        document.getElementById('metric-coherence').textContent = val.toFixed(4);
        document.getElementById('bar-coherence').style.width = (val * 100) + '%';
    }
    if (epistemic.tension_magnitude !== undefined) {
        const val = epistemic.tension_magnitude;
        document.getElementById('metric-tension').textContent = val.toFixed(4);
        document.getElementById('bar-tension').style.width = Math.min(val * 100, 100) + '%';
    }
    // Update ethical alignment if available
    if (epistemic.ethical_alignment !== undefined) {
        document.getElementById('metric-eta').textContent =
            epistemic.ethical_alignment.toFixed(3);
    } else if (epistemic.mean_coherence !== undefined) {
        // Fall back: derive eta from mean coherence as a proxy
        document.getElementById('metric-eta').textContent =
            epistemic.mean_coherence.toFixed(3);
    }
}

// ── Session Management ──
function newChat() {
    fetch('/api/session/new', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            // Clear chat
            const area = document.getElementById('chat-area');
            area.innerHTML = '';
            // Show welcome with starter cards
            const welcome = document.createElement('div');
            welcome.className = 'welcome';
            welcome.id = 'welcome';
            welcome.innerHTML = `
                <h2>What would you like to explore?</h2>
                <p>Codette routes your question to the best reasoning perspective automatically.</p>
                <div class="welcome-grid">
                    <div class="welcome-card" onclick="askQuestion('Explain why objects fall to the ground')">
                        <div class="welcome-card-title" style="color:var(--newton)">Newton</div>
                        <div class="welcome-card-desc">Explain why objects fall to the ground</div>
                    </div>
                    <div class="welcome-card" onclick="askQuestion('Design a creative solution for sustainable cities')">
                        <div class="welcome-card-title" style="color:var(--davinci)">DaVinci</div>
                        <div class="welcome-card-desc">Design a creative solution for sustainable cities</div>
                    </div>
                    <div class="welcome-card" onclick="askQuestion('How do I cope with feeling overwhelmed?')">
                        <div class="welcome-card-title" style="color:var(--empathy)">Empathy</div>
                        <div class="welcome-card-desc">How do I cope with feeling overwhelmed?</div>
                    </div>
                    <div class="welcome-card" onclick="askQuestion('What is consciousness and can AI have it?')">
                        <div class="welcome-card-title" style="color:var(--consciousness)">Consciousness</div>
                        <div class="welcome-card-desc">What is consciousness and can AI have it?</div>
                    </div>
                </div>
            `;
            area.appendChild(welcome);
            // Reset metrics
            document.getElementById('metric-coherence').textContent = '0.00';
            document.getElementById('metric-tension').textContent = '0.00';
            document.getElementById('metric-eta').textContent = '--';
            document.getElementById('bar-coherence').style.width = '0%';
            document.getElementById('bar-tension').style.width = '0%';
            document.getElementById('cocoon-attractors').textContent = '0';
            document.getElementById('cocoon-glyphs').textContent = '0';
            // Reset spiderweb
            if (spiderwebViz) {
                spiderwebViz._initDefaultState();
                spiderwebViz.coherence = 0;
                spiderwebViz.attractors = [];
            }
            loadSessions();
        });
}

function loadSessions() {
    fetch('/api/sessions')
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById('session-list');
            const sessions = data.sessions || [];
            document.getElementById('cocoon-sessions').textContent = sessions.length;

            list.innerHTML = sessions.map(s => `
                <div class="session-item" onclick="loadSession('${s.session_id}')"
                     title="${s.title}">
                    ${s.title || 'Untitled'}
                </div>
            `).join('');
        })
        .catch(() => {});
}

function loadSession(sessionId) {
    fetch('/api/session/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) return;

        // Clear and rebuild chat
        const area = document.getElementById('chat-area');
        area.innerHTML = '';

        (data.messages || []).forEach(msg => {
            addMessage(msg.role, msg.content, msg.metadata || {});
        });

        if (data.state) {
            updateCocoonUI(data.state);
        }
    })
    .catch(err => {
        console.log('Failed to load session:', err);
    });
}

// ── Utilities ──
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    // Lightweight markdown renderer — no dependencies
    let html = escapeHtml(text);

    // Code blocks: ```lang\n...\n```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
        '<pre class="code-block"><code>$2</code></pre>');

    // Inline code: `code`
    html = html.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');

    // Bold: **text** or __text__
    html = html.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__([^_\n]+?)__/g, '<strong>$1</strong>');

    // Headers: ### text (on its own line) — before bullets to avoid conflict
    html = html.replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>');
    html = html.replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>');
    html = html.replace(/^# (.+)$/gm, '<div class="md-h1">$1</div>');

    // Bullet lists: - item or * item — before italic to prevent * conflicts
    html = html.replace(/^[\-\*] (.+)$/gm, '<div class="md-li">$1</div>');

    // Numbered lists: 1. item
    html = html.replace(/^\d+\. (.+)$/gm, '<div class="md-li md-oli">$1</div>');

    // Italic: *text* or _text_ — AFTER bullets, restricted to single line
    html = html.replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, '<em>$1</em>');
    html = html.replace(/(?<!\w)_([^_\n]+?)_(?!\w)/g, '<em>$1</em>');

    // Line breaks (preserve double newlines as paragraph breaks)
    html = html.replace(/\n\n/g, '<br><br>');
    html = html.replace(/\n/g, '<br>');

    return html;
}
