/**
 * Habits Module - Manages the Habit Streak Tracker UI, Logic, and Persistence
 * Fully redesigned to month-calendar board with robust cyclic tracking statuses.
 */

const PRESET_COLORS = [
    'var(--primary-color)',
    '#e74c3c', // Red
    '#3498db', // Blue
    '#f1c40f', // Yellow
    '#9b59b6'  // Purple
];

const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

const Habits = {
    init() {
        this.board = document.getElementById('habits-board');
        this.addBtn = document.getElementById('add-habit-btn');
        if (!this.board) return;

        this.todayDate = new Date();
        // Centralized view month for the calendar rendering
        this.viewMonthDate = new Date(); 

        this.injectModal();

        this.modal = document.getElementById('add-habit-modal');
        this.closeModalBtn = document.getElementById('close-add-habit-modal');
        this.saveBtn = document.getElementById('save-new-habit-btn');
        this.titleInput = document.getElementById('new-habit-title');
        this.typeInput = document.getElementById('new-habit-type');
        
        this.selectedColor = PRESET_COLORS[0];

        this.bindEvents();
        this.migrateLegacyLogs();
        this.renderHabits();
    },

    injectModal() {
        if (document.getElementById('add-habit-modal')) return;

        const modalHtml = `
            <div id="add-habit-modal" class="popup">
                <div class="popup-header">
                    <h2>📌 Create New Habit sticky note</h2>
                    <button type="button" id="close-add-habit-modal" class="close-btn">×</button>
                </div>
                <div>
                    <label style="opacity: 0.8; margin-bottom: 5px; display: block;">Habit Title</label>
                    <input type="text" id="new-habit-title" placeholder="e.g. Stop Gaming, No Sugar, Read 10 Pages" class="journal-input" maxlength="50" style="width: 100%; box-sizing: border-box; padding: 10px; border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid var(--border-color); color: var(--text-color);">
                    
                    <label style="opacity: 0.8; margin-top: 15px; display: block;">Trigger Type</label>
                    <select id="new-habit-type" class="journal-input" style="width: 100%; box-sizing: border-box; padding: 10px; border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid var(--border-color); color: var(--text-color);">
                        <option value="build">✅ Build (Target: Do it)</option>
                        <option value="avoid">🚫 Avoid (Target: Don't do it)</option>
                    </select>

                    <label style="opacity: 0.8; margin-top: 15px; display: block;">Accent Color</label>
                    <div class="habit-color-picker" id="habit-color-picker"></div>

                    <div style="margin-top: 25px; text-align: right; display: flex; justify-content: flex-end; gap: 10px;">
                        <button type="button" class="secondary-btn" id="cancel-add-habit-btn" style="background: transparent; border: 1px solid var(--border-color); color: var(--text-muted);">Cancel</button>
                        <button type="button" id="save-new-habit-btn" class="primary-btn">📌 Pin It</button>
                    </div>
                </div>
            </div>
        `;

        const wrapper = document.createElement('div');
        wrapper.innerHTML = modalHtml;
        document.body.appendChild(wrapper.firstElementChild);

        const picker = document.getElementById('habit-color-picker');
        PRESET_COLORS.forEach((color, idx) => {
            const circle = document.createElement('div');
            circle.className = `color-option ${idx === 0 ? 'selected' : ''}`;
            circle.style.backgroundColor = color === 'var(--primary-color)' ? '#ff3232' : color;
            circle.dataset.color = color;
            
            circle.addEventListener('click', () => {
                document.querySelectorAll('.color-option').forEach(c => c.classList.remove('selected'));
                circle.classList.add('selected');
                this.selectedColor = color;
            });
            picker.appendChild(circle);
        });
    },

    bindEvents() {
        if(this.addBtn) {
            this.addBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.titleInput.value = '';
                this.typeInput.value = 'build';
                this.selectedColor = PRESET_COLORS[0];
                document.querySelectorAll('.color-option').forEach((c, idx) => {
                    c.classList.toggle('selected', idx === 0);
                });
                this.modal.classList.add('open');
                const overlay = document.getElementById('popup-overlay');
                if (overlay) overlay.classList.add('active');
                setTimeout(() => this.titleInput.focus(), 100);
            });
        }

        if(this.closeModalBtn) {
            this.closeModalBtn.addEventListener('click', (e) => { e.preventDefault(); this.closeModal(); });
        }

        const cancelBtn = document.getElementById('cancel-add-habit-btn');
        if(cancelBtn) {
            cancelBtn.addEventListener('click', (e) => { e.preventDefault(); this.closeModal(); });
        }
        
        if(this.saveBtn) {
            this.saveBtn.addEventListener('click', (e) => {
                e.preventDefault();
                const title = this.titleInput.value.trim();
                const triggerType = this.typeInput.value;
                if (title) {
                    const newHabit = {
                        id: 'habit_' + Date.now().toString(),
                        title: title,
                        accentColor: this.selectedColor,
                        triggerType: triggerType,
                        createdAt: this.formatDate(new Date()),
                        dailyLogs: []
                    };
                    Storage.addHabit(newHabit);
                    this.closeModal();
                    this.renderHabits();
                } else {
                    if (typeof Toast !== 'undefined') Toast.warning('Please enter a habit title.');
                    else alert('Please enter a habit title.');
                }
            });
        }

        if(this.titleInput) {
            this.titleInput.addEventListener('keyup', (e) => {
                if (e.key === 'Enter') this.saveBtn.click();
            });
        }
    },

    closeModal() {
        if(this.modal) this.modal.classList.remove('open');
        const overlay = document.getElementById('popup-overlay');
        if (overlay) overlay.classList.remove('active');
    },

    formatDate(dateObj) {
        const offset = dateObj.getTimezoneOffset();
        const localDate = new Date(dateObj.getTime() - (offset*60*1000));
        return localDate.toISOString().split('T')[0];
    },

    // Ensure backwards compatibility with original boolean logs
    migrateLegacyLogs() {
        let habits = Storage.getHabits() || [];
        let modified = false;
        habits.forEach(habit => {
            if (!habit.triggerType) {
                habit.triggerType = 'build';
                modified = true;
            }
            if (habit.dailyLogs) {
                habit.dailyLogs.forEach(log => {
                    if (log.status === undefined && log.checked !== undefined) {
                        log.status = log.checked ? 'success' : 'empty';
                        modified = true;
                    }
                });
            }
        });
        if (modified && typeof Storage.save === 'function') {
            Storage.save();
        }
    },

    calculateStreak(habit) {
        let streak = 0;
        let checkDate = new Date();
        const todayStr = this.formatDate(checkDate);
        
        const logMap = {};
        if (habit.dailyLogs) {
            habit.dailyLogs.forEach(l => {
                logMap[l.date] = l.status;
            });
        }

        const todayStatus = logMap[todayStr] || 'empty';
        if (todayStatus === 'success') {
            streak++;
        } else if (todayStatus === 'failed') {
            return 0; // broken
        }

        checkDate.setDate(checkDate.getDate() - 1);
        
        while (true) {
            const dateStr = this.formatDate(checkDate);
            const status = logMap[dateStr] || 'empty';
            
            // Empty in the past breaks the streak. Failed breaks it.
            if (status === 'success') {
                streak++;
            } else {
                break;
            }
            checkDate.setDate(checkDate.getDate() - 1);
        }
        return streak;
    },

    getTodayLog(habit) {
        const todayStr = this.formatDate(new Date());
        return habit.dailyLogs.find(l => l.date === todayStr);
    },

    updateNote(id, noteText) {
        const habits = Storage.getHabits();
        const habitIndex = habits.findIndex(h => h.id === id);
        if (habitIndex !== -1) {
            const habit = habits[habitIndex];
            const todayStr = this.formatDate(new Date());
            
            let logIndex = habit.dailyLogs.findIndex(l => l.date === todayStr);
            if (logIndex !== -1) {
                habit.dailyLogs[logIndex].note = noteText;
            } else {
                habit.dailyLogs.push({ date: todayStr, status: 'empty', note: noteText });
            }
            Storage.updateHabit(id, { dailyLogs: habit.dailyLogs });
        }
    },

    deleteHabit(id) {
        if (confirm('Delete this habit permanently?')) {
            Storage.deleteHabit(id);
            this.renderHabits();
        }
    },
    
    changeMonth(step) {
        this.viewMonthDate.setMonth(this.viewMonthDate.getMonth() + step);
        this.renderHabits();
    },

    renderCalendarGrid(habit) {
        const year = this.viewMonthDate.getFullYear();
        const month = this.viewMonthDate.getMonth();
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        
        let container = document.createElement('div');
        
        let successCount = 0;
        let failedCount = 0;
        let emptyCount = 0;
        
        let gridHtml = '<div class="habit-calendar-grid">';

        for (let day = 1; day <= daysInMonth; day++) {
            const dateObj = new Date(year, month, day);
            const dateStr = this.formatDate(dateObj);
            const todayStr = this.formatDate(this.todayDate);

            const log = habit.dailyLogs.find(l => l.date === dateStr);
            let status = log ? log.status : 'empty';

            let tileClasses = ['habit-day-tile', status];
            if (dateStr === todayStr) tileClasses.push('today');
            if (dateStr > todayStr) tileClasses.push('future');

            if (status === 'success') successCount++;
            else if (status === 'failed') failedCount++;
            else if (dateStr <= todayStr && dateStr >= habit.createdAt) emptyCount++; 

            gridHtml += `
                <div class="${tileClasses.join(' ')}" data-date="${dateStr}" title="${dateStr}">
                    ${day}
                </div>
            `;
        }
        
        gridHtml += '</div>';
        
        const summaryHtml = `
            <div class="habit-month-summary">
                <span style="color:#4caf50;">✅ ${successCount} kept</span>
                <span style="color:#c0392b;">❌ ${failedCount} failed</span>
                <span style="opacity:0.6;">⬜ ${emptyCount} remaining</span>
            </div>
        `;

        container.innerHTML = gridHtml + summaryHtml;

        // Add Cycle Clicking Events
        container.querySelectorAll('.habit-day-tile:not(.future)').forEach(tile => {
            tile.addEventListener('click', () => {
                const dateStr = tile.dataset.date;
                const habits = Storage.getHabits();
                const hIndex = habits.findIndex(h => h.id === habit.id);
                if (hIndex === -1) return;
                
                const targetHabit = habits[hIndex];
                let lIndex = targetHabit.dailyLogs.findIndex(l => l.date === dateStr);
                
                if (lIndex === -1) {
                    // Empty -> Success
                    targetHabit.dailyLogs.push({ date: dateStr, status: 'success', note: '' });
                } else {
                    const currentStatus = targetHabit.dailyLogs[lIndex].status;
                    if (currentStatus === 'success') {
                        targetHabit.dailyLogs[lIndex].status = 'failed';
                    } else if (currentStatus === 'failed') {
                        targetHabit.dailyLogs[lIndex].status = 'empty';
                    } else {
                        targetHabit.dailyLogs[lIndex].status = 'success';
                    }
                }
                
                Storage.updateHabit(targetHabit.id, { dailyLogs: targetHabit.dailyLogs });
                this.renderHabits(); // Re-render everything to update streak accurately
            });
        });

        return container;
    },

    renderHabits() {
        const habits = Storage.getHabits() || [];
        this.board.innerHTML = '';

        if (habits.length === 0) {
            this.board.innerHTML = `
                <div style="width: 100%; text-align: center; margin-top: 50px; opacity: 0.6; grid-column: 1 / -1;">
                    <div style="font-size: 4rem; margin-bottom: 20px;">📌</div>
                    <h2>No habits pinned yet.</h2>
                </div>
            `;
            return;
        }

        const monthName = MONTH_NAMES[this.viewMonthDate.getMonth()];
        const yearYear = this.viewMonthDate.getFullYear();

        habits.forEach(habit => {
            const streak = this.calculateStreak(habit);
            const todayLog = this.getTodayLog(habit);
            const currentNote = todayLog ? todayLog.note : "";
            const startDateDisplay = new Date(habit.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
            
            const badgeType = habit.triggerType === 'avoid' ? '🚫 Avoid' : '✅ Build';

            const card = document.createElement('div');
            card.className = 'habit-card';
            if (habit.accentColor) {
                card.style.borderTopColor = habit.accentColor;
            }

            const noteId = `note-${habit.id}`;

            let htmlStart = `
                <div class="habit-pin">📌</div>
                
                <div class="habit-card-header">
                    <div class="habit-title-container">
                        <span class="habit-type-badge">${badgeType}</span>
                        <div class="habit-card-title">${habit.title}</div>
                    </div>
                </div>
                
                <div class="habit-streak-counter" ${habit.accentColor && habit.accentColor !== 'var(--primary-color)' ? `style="color: ${habit.accentColor}"` : ''}>🔥 Day ${streak}</div>

                <div class="habit-month-nav">
                    <button type="button" class="btn-prev-month">←</button>
                    <span>${monthName} ${yearYear}</span>
                    <button type="button" class="btn-next-month">→</button>
                </div>
            `;
            
            let htmlEnd = `
                <textarea id="${noteId}" class="habit-scratch" rows="3" placeholder="Scratch note for today...">${currentNote}</textarea>

                <div class="habit-footer" style="margin-top:15px;">Started: ${startDateDisplay}</div>
                <button type="button" class="habit-delete-btn" title="Delete Habit">✕</button>
            `;

            card.innerHTML = htmlStart;
            
            const calGrid = this.renderCalendarGrid(habit);
            card.appendChild(calGrid);
            
            const elEnd = document.createElement('div');
            elEnd.innerHTML = htmlEnd;
            while(elEnd.firstChild) { card.appendChild(elEnd.firstChild); }

            // Events
            card.querySelector('.btn-prev-month').addEventListener('click', (e) => { e.preventDefault(); this.changeMonth(-1); });
            card.querySelector('.btn-next-month').addEventListener('click', (e) => { e.preventDefault(); this.changeMonth(1); });

            const scratchInput = card.querySelector(`#${noteId}`);
            if (scratchInput) {
                scratchInput.addEventListener('blur', (e) => {
                    this.updateNote(habit.id, e.target.value);
                });
            }

            const delBtn = card.querySelector('.habit-delete-btn');
            if (delBtn) {
                delBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.deleteHabit(habit.id);
                });
            }

            this.board.appendChild(card);
        });
    }
};

window.Habits = Habits;
