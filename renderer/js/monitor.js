/**
 * Mental State Monitor — frontend logic
 *
 * Reads theme from localStorage so the page matches whatever theme the user set
 * in the main journal app without needing the full Storage module.
 */

(function () {
    'use strict';

    // ── Apply saved theme / neon / pattern ─────────────────────────────────
    (function applyTheme() {
        try {
            const raw = localStorage.getItem('daily_productivity_full_data');
            if (!raw) return;
            const data = JSON.parse(raw);
            const theme   = data?.settings?.theme   || 'default';
            const neon    = data?.settings?.neon    || 'none';
            const pattern = data?.settings?.pattern || 'none';
            if (window.Themes) {
                Themes.apply(theme);
                Themes.applyNeon(neon);
                Themes.applyPattern(pattern);
            }
        } catch (_) {}
    })();

    // ── State ───────────────────────────────────────────────────────────────
    const S = {
        session:    null,   // active session object
        currentQ:  0,       // index into session.questions
        answers:   {},      // { questionId: { text, skipped } }
        prevScreen: 'welcome',
    };

    // ── Screen management ───────────────────────────────────────────────────
    const SCREENS = ['welcome', 'question', 'complete', 'insights', 'history', 'detail'];

    function show(name) {
        SCREENS.forEach(n => {
            const el = document.getElementById(`screen-${n}`);
            if (el) el.classList.toggle('hidden', n !== name);
        });
        S.prevScreen = name;
    }

    // ── Mood slider helpers ─────────────────────────────────────────────────
    function bindSlider(sliderId, displayId) {
        const slider  = document.getElementById(sliderId);
        const display = document.getElementById(displayId);
        if (!slider || !display) return;
        slider.addEventListener('input', () => {
            display.textContent = parseFloat(slider.value).toFixed(1);
        });
    }

    // ── Mood strip HTML helper ──────────────────────────────────────────────
    function moodStrip(moodStart, moodEnd, topicsEngaged, topicsSkipped) {
        const delta     = moodEnd !== null && moodStart !== null ? moodEnd - moodStart : null;
        const deltaStr  = delta !== null
            ? `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`
            : '—';
        const deltaColor = delta !== null ? (delta >= 0 ? '#34d399' : '#f87171') : 'inherit';

        return `
            <div class="mood-cell">
                <div class="mood-cell-lbl">Start</div>
                <div class="mood-cell-val">${moodStart !== null ? moodStart.toFixed(1) : '?'}</div>
            </div>
            <div class="mood-cell">
                <div class="mood-cell-lbl">End</div>
                <div class="mood-cell-val">${moodEnd !== null ? moodEnd.toFixed(1) : '?'}</div>
            </div>
            <div class="mood-cell">
                <div class="mood-cell-lbl">Change</div>
                <div class="mood-cell-val" style="color:${deltaColor}">${deltaStr}</div>
            </div>
            <div class="mood-cell">
                <div class="mood-cell-lbl">Engaged</div>
                <div class="mood-cell-val">${(topicsEngaged || []).length}</div>
            </div>
            <div class="mood-cell">
                <div class="mood-cell-lbl">Skipped</div>
                <div class="mood-cell-val">${(topicsSkipped || []).length}</div>
            </div>`;
    }

    // ── Start session (two-phase: tune → questions) ─────────────────────────
    async function startSession() {
        const moodStart = parseFloat(document.getElementById('start-mood').value);
        document.getElementById('start-btn').classList.add('hidden');
        document.getElementById('start-loading').classList.remove('hidden');

        // Animate step indicators while the two LLM phases run server-side
        const stepDefs = [
            { id: 'step-1', label: '📚 Reading all journal entries…',             delay: 0    },
            { id: 'step-2', label: '🔍 Phase 1: Building psychological profile…', delay: 4000 },
            { id: 'step-3', label: '🎯 Phase 2: Generating targeted questions…',  delay: 9000 },
        ];
        const timers = [];
        const msgEl  = document.getElementById('start-loading-msg');
        stepDefs.forEach(({ id, label, delay }) => {
            const t = setTimeout(() => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = '1'; el.textContent = '⚡ ' + label.replace(/^[^ ]+ /, ''); }
                if (msgEl) msgEl.textContent = label.replace(/^[^ ]+ /, '');
            }, delay);
            timers.push(t);
        });

        try {
            const res  = await fetch('/api/monitor/session/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mood_start: moodStart }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error || 'Failed to start session');

            // Mark all steps complete
            timers.forEach(t => clearTimeout(t));
            stepDefs.forEach(({ id, label }) => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = '1'; el.textContent = '✅ ' + label.replace(/^[^ ]+ /, ''); }
            });

            S.session  = data.session;
            S.currentQ = 0;
            S.answers  = {};

            // Show brief tuning summary banner on question screen
            if (data.tuning_profile) showTuningBanner(data.tuning_profile, data.dedup_count || 0);

            renderQuestion();
            show('question');
        } catch (err) {
            timers.forEach(t => clearTimeout(t));
            alert('Could not start session: ' + err.message);
        } finally {
            document.getElementById('start-btn').classList.remove('hidden');
            document.getElementById('start-loading').classList.add('hidden');
            // Reset steps for next time
            stepDefs.forEach(({ id, label }, i) => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = i === 0 ? '1' : '0.3'; el.textContent = '⏳ ' + label.replace(/^[^ ]+ /, ''); }
            });
            if (msgEl) msgEl.textContent = 'Reading journal entries…';
        }
    }

    // ── Tuning profile banner shown at top of question screen ───────────────
    function showTuningBanner(profile, dedupCount) {
        const existing = document.getElementById('tune-banner');
        if (existing) existing.remove();

        const journals = profile.total_journals_read || '?';
        const themes   = (profile.dominant_themes   || []).slice(0, 3).join(', ') || '—';
        const tensions = (profile.unresolved_tensions || []).slice(0, 2).join(', ') || '—';

        const banner = document.createElement('div');
        banner.id = 'tune-banner';
        banner.style.cssText = [
            'background:rgba(16,185,129,.08)',
            'border:1px solid rgba(16,185,129,.25)',
            'border-radius:12px',
            'padding:12px 16px',
            'margin-bottom:16px',
            'font-size:.8rem',
            'line-height:1.6',
        ].join(';');
        const dedupNote = dedupCount > 0
            ? `<br><span style="opacity:.5;font-size:.75rem;">⟳ ${dedupCount} question(s) regenerated to avoid repetition</span>`
            : '';
        banner.innerHTML =
            `<strong style="color:#34d399;">✅ Tuning complete</strong> — read <strong>${journals}</strong> journal entries<br>` +
            `<span style="opacity:.65;">Key themes: <em>${themes}</em></span><br>` +
            `<span style="opacity:.65;">Unresolved tensions: <em>${tensions}</em></span>` +
            dedupNote;

        const qScreen = document.getElementById('screen-question');
        if (qScreen) qScreen.insertBefore(banner, qScreen.firstChild);

        setTimeout(() => {
            if (banner.parentNode) {
                banner.style.transition = 'opacity 1.2s';
                banner.style.opacity    = '0';
                setTimeout(() => banner.remove(), 1200);
            }
        }, 10000);
    }

    // ── Render current question ─────────────────────────────────────────────
    function renderQuestion() {
        const questions = S.session.questions;
        const q         = questions[S.currentQ];
        if (!q) { showComplete(); return; }

        // Progress dots
        const dotsEl = document.getElementById('q-dots');
        dotsEl.innerHTML = '';
        questions.forEach((qi, i) => {
            const dot = document.createElement('div');
            dot.className = 'q-dot';
            if (i < S.currentQ) {
                dot.classList.add(S.answers[questions[i].id]?.skipped ? 'skipped' : 'done');
            } else if (i === S.currentQ) {
                dot.classList.add('active');
            }
            dotsEl.appendChild(dot);
        });

        // Badge
        const badge = document.getElementById('q-badge');
        badge.className = `cat-badge ${q.category || 'general'}`;
        badge.textContent = (q.category || 'general').replace(/_/g, ' ');

        // Text
        document.getElementById('q-text').textContent = q.text;

        // Restore saved answer
        const prev = S.answers[q.id];
        document.getElementById('q-ans').value = prev?.text || '';

        // Last question label
        const isLast = S.currentQ === questions.length - 1;
        document.getElementById('next-btn').textContent = isLast ? 'Finish →' : 'Next →';
    }

    function showComplete() {
        // Pre-set end mood to current start mood as a sensible default
        const startVal = document.getElementById('start-mood').value;
        document.getElementById('end-mood').value        = startVal;
        document.getElementById('end-mood-val').textContent = parseFloat(startVal).toFixed(1);
        show('complete');
    }

    // ── Submit answer ───────────────────────────────────────────────────────
    async function submitAnswer(skipped) {
        const q    = S.session.questions[S.currentQ];
        const text = document.getElementById('q-ans').value.trim();

        if (!skipped && !text) {
            const ta = document.getElementById('q-ans');
            ta.focus();
            ta.style.borderColor = 'var(--primary-color)';
            setTimeout(() => { ta.style.borderColor = ''; }, 1600);
            return;
        }

        S.answers[q.id] = { text: skipped ? '' : text, skipped };

        // Fire-and-forget save (non-blocking)
        fetch('/api/monitor/session/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id:  S.session.session_id,
                question_id: q.id,
                text:        skipped ? '' : text,
                skipped,
            }),
        }).catch(() => {});

        S.currentQ++;
        if (S.currentQ >= S.session.questions.length) {
            showComplete();
        } else {
            document.getElementById('q-ans').value = '';
            renderQuestion();
        }
    }

    // ── Complete session (3-phase pipeline) ────────────────────────────────
    async function completeSession() {
        const moodEnd = parseFloat(document.getElementById('end-mood').value);
        document.getElementById('complete-btn').classList.add('hidden');
        document.getElementById('complete-loading').classList.remove('hidden');

        // Animate 3 pipeline steps
        const cSteps = [
            { id: 'c-step-1', label: 'Phase 1: Cross-referencing answers with all journals…', delay: 0    },
            { id: 'c-step-2', label: 'Phase 2: Enriching from psychology datasets…',          delay: 5000 },
            { id: 'c-step-3', label: 'Phase 3: Generating final insights…',                   delay: 10000 },
        ];
        const timers = [];
        const msgEl = document.getElementById('complete-loading-msg');
        cSteps.forEach(({ id, label, delay }) => {
            const t = setTimeout(() => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = '1'; el.textContent = '⚡ ' + label; }
                if (msgEl) msgEl.textContent = label;
            }, delay);
            timers.push(t);
        });

        try {
            const res  = await fetch('/api/monitor/session/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: S.session.session_id, mood_end: moodEnd }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error || 'Failed to complete');

            timers.forEach(t => clearTimeout(t));
            cSteps.forEach(({ id, label }) => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = '1'; el.textContent = '✅ ' + label; }
            });

            S.session = data.session;
            renderInsights(data.session, data.insights, data.journal_inferences);
            show('insights');
            loadSidebar();
        } catch (err) {
            timers.forEach(t => clearTimeout(t));
            alert('Error: ' + err.message);
        } finally {
            document.getElementById('complete-btn').classList.remove('hidden');
            document.getElementById('complete-loading').classList.add('hidden');
            // Reset step labels for next time
            cSteps.forEach(({ id, label }, i) => {
                const el = document.getElementById(id);
                if (el) { el.style.opacity = i === 0 ? '1' : '0.3'; el.textContent = '⏳ ' + label; }
            });
            if (msgEl) msgEl.textContent = 'Comparing answers to journal history…';
        }
    }

    // ── Render insights + journal inferences ────────────────────────────────
    function renderInsights(session, insights, inferences) {
        document.getElementById('insights-date').textContent = session.date || '';
        document.getElementById('insights-strip').innerHTML  = moodStrip(
            session.mood_start, session.mood_end,
            session.topics_engaged, session.topics_skipped
        );

        // Journal cross-reference inference block
        const infBlock = document.getElementById('inference-block');
        const inf = inferences || session.journal_inferences;
        if (inf && inf.key_inference) {
            document.getElementById('inference-key').textContent = '"' + inf.key_inference + '"';
            const parts = [];
            if ((inf.confirmed_patterns || []).length)  parts.push('✔ Confirmed: '   + inf.confirmed_patterns.join('; '));
            if ((inf.new_revelations   || []).length)   parts.push('💡 Revealed: '    + inf.new_revelations.join('; '));
            if ((inf.contradictions    || []).length)   parts.push('⚡ Contradicted: ' + inf.contradictions.join('; '));
            if ((inf.progression       || []).length)   parts.push('📈 Progression: '  + inf.progression.join('; '));
            document.getElementById('inference-details').textContent = parts.join('\n');
            infBlock.classList.remove('hidden');
        } else {
            infBlock.classList.add('hidden');
        }

        document.getElementById('insights-body').textContent = insights || session.insights || '';
    }

    // ── Load sidebar ────────────────────────────────────────────────────────
    async function loadSidebar() {
        try {
            const res  = await fetch('/api/monitor/progress');
            const data = await res.json();
            if (!data.ok) return;
            const p     = data.progress;
            const stats = p.overall_stats || {};

            document.getElementById('s-total').textContent = stats.sessions_completed || 0;

            const avg = stats.avg_mood_improvement;
            if (avg !== undefined) {
                const sign = avg >= 0 ? '+' : '';
                const el   = document.getElementById('s-avg');
                el.textContent = `${sign}${avg.toFixed(2)}`;
                el.style.color = avg >= 0 ? '#34d399' : '#f87171';
            }
            document.getElementById('s-last').textContent = p.last_updated || '—';

            // Topic bars
            const barsEl = document.getElementById('topic-bars');
            const topics  = Object.entries(p.topics || {})
                .sort((a, b) => b[1].progress_score - a[1].progress_score);

            if (topics.length === 0) {
                barsEl.innerHTML = '<p style="opacity:.4;font-size:.82rem;">Complete a session to track topics.</p>';
            } else {
                barsEl.innerHTML = topics.map(([key, t]) => {
                    const pct   = Math.round((t.progress_score || 0) * 100);
                    const label = t.label || key.replace(/_/g, ' ');
                    return `
                        <div class="topic-row">
                            <div class="topic-hdr">
                                <span class="topic-name">${label}</span>
                                <span class="topic-pct">${pct}%</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width:${pct}%"></div>
                            </div>
                        </div>`;
                }).join('');
            }
        } catch (_) {}
    }

    // ── Load history ────────────────────────────────────────────────────────
    async function loadHistory() {
        const listEl  = document.getElementById('history-list');
        const noEl    = document.getElementById('no-history');
        listEl.innerHTML = '<div class="loading-blk"><div class="spinner"></div></div>';
        noEl.classList.add('hidden');

        try {
            const res  = await fetch('/api/monitor/sessions');
            const data = await res.json();
            if (!data.ok) throw new Error();

            const sessions = data.sessions;
            if (!sessions.length) {
                listEl.innerHTML = '';
                noEl.classList.remove('hidden');
                return;
            }

            listEl.innerHTML = sessions.map(s => {
                const delta     = (s.mood_end !== null && s.mood_start !== null)
                    ? s.mood_end - s.mood_start : null;
                const deltaStr  = delta !== null
                    ? `mood ${delta >= 0 ? '+' : ''}${delta.toFixed(1)}` : '';
                const deltaColor = delta !== null
                    ? (delta >= 0 ? '#34d399' : '#f87171') : '';

                const engChips  = (s.topics_engaged || [])
                    .map(t => `<span class="chip">${t.replace(/_/g, ' ')}</span>`).join('');
                const skipChips = (s.topics_skipped || [])
                    .map(t => `<span class="chip skipped">${t.replace(/_/g, ' ')}</span>`).join('');

                return `
                    <div class="hist-card" data-id="${s.session_id}">
                        <div class="hist-date">
                            ${s.date}
                            <span style="opacity:.35;font-weight:400;font-size:.78rem;"> · ${s.status}</span>
                        </div>
                        <div class="hist-meta">
                            <span>📝 ${s.answers_count}/${s.questions_count} answered</span>
                            ${deltaStr ? `<span style="color:${deltaColor}">${deltaStr}</span>` : ''}
                        </div>
                        <div class="hist-chips">${engChips}${skipChips}</div>
                        ${s.insights_preview
                            ? `<div class="hist-preview">${s.insights_preview}</div>`
                            : ''}
                    </div>`;
            }).join('');

            listEl.querySelectorAll('.hist-card').forEach(card => {
                card.addEventListener('click', () => loadDetail(card.dataset.id));
            });
        } catch (_) {
            listEl.innerHTML = '<p style="opacity:.45;text-align:center;padding:24px;">Failed to load history.</p>';
        }
    }

    // ── Load session detail ─────────────────────────────────────────────────
    async function loadDetail(sessionId) {
        try {
            const res  = await fetch(`/api/monitor/session/${sessionId}`);
            const data = await res.json();
            if (!data.ok) throw new Error();
            const s = data.session;

            document.getElementById('detail-title').textContent = `Session — ${s.date}`;
            document.getElementById('detail-strip').innerHTML   = moodStrip(
                s.mood_start, s.mood_end, s.topics_engaged, s.topics_skipped
            );

            document.getElementById('detail-qa').innerHTML = (s.questions || []).map(q => {
                const ans     = (s.answers || []).find(a => a.question_id === q.id);
                const skipped = !ans || ans.skipped;
                return `
                    <div class="qa-block ${skipped ? 'skipped' : ''}">
                        <div class="cat-badge ${q.category || ''}" style="display:inline-block;margin-bottom:6px;">
                            ${(q.category || '').replace(/_/g, ' ')}
                        </div>
                        <div class="qa-q">${q.text}</div>
                        <div class="qa-a">${skipped ? '<em>Skipped</em>' : (ans.text || '')}</div>
                    </div>`;
            }).join('');

            const insightEl = document.getElementById('detail-insights');
            insightEl.innerHTML = s.insights
                ? `<h4 style="margin-bottom:10px;opacity:.65;">🔍 Insights</h4>
                   <p class="insights-text">${s.insights}</p>`
                : '';

            show('detail');
        } catch (_) {
            alert('Could not load session.');
        }
    }

    // ── Wire up buttons ─────────────────────────────────────────────────────
    function bindButtons() {
        document.getElementById('start-btn').addEventListener('click', startSession);
        document.getElementById('next-btn').addEventListener('click',  () => submitAnswer(false));
        document.getElementById('skip-btn').addEventListener('click',  () => submitAnswer(true));
        document.getElementById('complete-btn').addEventListener('click', completeSession);
        document.getElementById('new-session-btn').addEventListener('click', () => show('welcome'));
        document.getElementById('to-history-btn').addEventListener('click',  () => { loadHistory(); show('history'); });
        document.getElementById('history-btn').addEventListener('click',     () => { loadHistory(); show('history'); });
        document.getElementById('hist-back-btn').addEventListener('click',   () => show(S.session ? 'insights' : 'welcome'));
        document.getElementById('detail-back-btn').addEventListener('click', () => show('history'));
    }

    // ── Init ────────────────────────────────────────────────────────────────
    function init() {
        bindSlider('start-mood', 'start-mood-val');
        bindSlider('end-mood',   'end-mood-val');
        bindButtons();
        loadSidebar();
    }

    document.addEventListener('DOMContentLoaded', init);
})();
