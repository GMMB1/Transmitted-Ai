/**
 * Weekly Reports Module
 */

const Weekly = {
    popup: null,
    listPopup: null,
    autoSaveInterval: null,
    isGenerating: false,

    /**
     * Initialize weekly module
     */
    init() {
        this.popup = document.getElementById('weekly-popup');
        this.listPopup = document.getElementById('weekly-list-popup');
        this.setupListeners();
    },

    /**
     * Setup event listeners
     */
    setupListeners() {
        // Open/close weekly popup - Now with 3 options: Cancel, Weekly Report, Analyze
        document.getElementById('weekly-report-btn').addEventListener('click', () => {
            // Create custom popup with 3 buttons
            const overlay = document.createElement('div');
            overlay.className = 'custom-alert-overlay active';
            overlay.id = 'weekly-choice-overlay';
            overlay.innerHTML = `
                <div class="custom-alert-box" style="width: 420px; text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 20px;">❓</div>
                    <h3 style="color: var(--accent-color); margin-bottom: 15px;">Generate Weekly Report</h3>
                    <p style="opacity: 0.8; margin-bottom: 25px;">Let Rona generate your Weekly Report?</p>
                    <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
                        <button id="cancel-weekly-choice" class="secondary-btn">Cancel</button>
                        <button id="confirm-weekly-choice" class="primary-btn">Yes</button>
                        <button id="analyze-choice" class="primary-btn" style="background: linear-gradient(135deg, #a855f7, #6366f1);">🧠 Analyze</button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);
            
            // Cancel
            document.getElementById('cancel-weekly-choice').addEventListener('click', () => {
                document.body.removeChild(overlay);
            });
            
            // Yes - Generate Weekly Report
            document.getElementById('confirm-weekly-choice').addEventListener('click', () => {
                document.body.removeChild(overlay);
                this.generateReport();
            });
            
            // Analyze - Open Analysis Modal
            document.getElementById('analyze-choice').addEventListener('click', () => {
                document.body.removeChild(overlay);
                App.showAnalysisDateSelectionModal();
            });
        });
        
        document.getElementById('close-weekly-popup').addEventListener('click', () => this.closeEditor());

        // Week data loading
        document.getElementById('report-month').addEventListener('change', () => this.loadWeekData());
        document.getElementById('report-week').addEventListener('change', () => this.loadWeekData());

        // Save weekly report
        document.getElementById('save-weekly-btn').addEventListener('click', () => this.save());

        // History buttons
        document.getElementById('weekly-history-btn').addEventListener('click', () => this.showList());
        document.getElementById('show-weekly-list-btn').addEventListener('click', () => this.showList());
        document.getElementById('close-weekly-list').addEventListener('click', () => this.hideList());
        document.getElementById('export-weekly-btn').addEventListener('click', () => this.exportCSV());

        // Panel focus
        document.getElementById('write-panel').addEventListener('click', () => this.activatePanel('write'));
        document.getElementById('read-panel').addEventListener('click', () => this.activatePanel('read'));

        // Back button in full day view
        document.getElementById('back-to-list-btn').addEventListener('click', () => {
            document.getElementById('full-day-view').classList.add('hidden');
        });

        // Weekly rating slider
        document.getElementById('weekly-rating-slider').addEventListener('input', (e) => {
            this.updateWeeklyEmoji(parseFloat(e.target.value));
        });

        // Year selector change
        document.getElementById('report-year')?.addEventListener('change', () => this.loadWeekData());

        // Create New button in history
        document.getElementById('create-new-weekly-btn')?.addEventListener('click', () => {
            this.hideList();
            this.openEditor();
        });

        // Populate years
        this.populateYears();
    },

    /**
     * Populate year dropdown (current year ±5)
     */
    populateYears() {
        const yearSelect = document.getElementById('report-year');
        if (!yearSelect) return;

        const currentYear = new Date().getFullYear();
        yearSelect.innerHTML = '';

        for (let year = currentYear - 5; year <= currentYear + 5; year++) {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            if (year === currentYear) option.selected = true;
            yearSelect.appendChild(option);
        }
    },

    /**
     * Open weekly editor
     */
    openEditor() {
        const now = new Date();
        const currentMonth = now.getMonth() + 1;
        let weekNumber = Math.ceil(now.getDate() / 7);
        if (weekNumber > 4) weekNumber = 4;

        document.getElementById('report-month').value = currentMonth;
        document.getElementById('report-week').value = weekNumber;

        this.popup.classList.add('open');
        this.loadWeekData();
        this.startAutoSave();
    },

    /**
     * Close weekly editor
     */
    closeEditor() {
        this.popup.classList.remove('open');
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
        }
    },

    /**
     * Start auto-save interval
     */
    startAutoSave() {
        if (this.autoSaveInterval) clearInterval(this.autoSaveInterval);

        this.autoSaveInterval = setInterval(() => {
            const title = document.getElementById('week-title').value;
            const details = document.getElementById('week-details').value;

            if (title.trim() || details.trim()) {
                localStorage.setItem('TempWeeklyReport', JSON.stringify({
                    month: document.getElementById('report-month').value,
                    week: document.getElementById('report-week').value,
                    title,
                    details,
                    rating: document.getElementById('weekly-rating-slider').value,
                    timestamp: Date.now()
                }));
            }
        }, 5000);
    },

    /**
     * Load week data
     */
    loadWeekData() {
        const year = parseInt(document.getElementById('report-year')?.value || new Date().getFullYear());
        const month = parseInt(document.getElementById('report-month').value);
        const week = parseInt(document.getElementById('report-week').value);

        if (!month || !week) return;
        const startDay = (week - 1) * 7 + 1;
        const endDay = Math.min(week * 7, 28);

        // Get journals for this week
        const journals = Storage.getJournals().filter(j => {
            const d = new Date(j.date);
            return d.getFullYear() === year &&
                (d.getMonth() + 1) === month &&
                d.getDate() >= startDay &&
                d.getDate() <= endDay;
        }).sort((a, b) => new Date(a.date) - new Date(b.date));

        // Render day cards
        const container = document.getElementById('days-summary-list');
        container.innerHTML = '';

        if (journals.length === 0) {
            container.innerHTML = '<p class="placeholder-msg">No journals found for this week.</p>';
        } else {
            journals.forEach((j, index) => {
                const cardWrapper = document.createElement('div');
                cardWrapper.className = 'day-card-wrapper';
                cardWrapper.dataset.index = index;

                const card = document.createElement('div');
                card.className = 'mini-day-card';
                card.innerHTML = `
                    <div class="mini-card-header">
                        <span class="mini-date">${j.date}</span>
                        <span class="mini-rating">${j.emoji || '⭐'} ${j.dailyRating}/10</span>
                    </div>
                    <div class="mini-title">${j.title}</div>
                `;
                card.addEventListener('click', () => this.toggleDayDetails(cardWrapper, j));
                cardWrapper.appendChild(card);

                // Create inline details panel (hidden by default)
                const details = document.createElement('div');
                details.className = 'inline-day-details';
                details.innerHTML = `
                    <div class="details-content">${j.details}</div>
                    <button class="collapse-btn">▲ Close</button>
                `;
                details.querySelector('.collapse-btn').addEventListener('click', (e) => {
                    e.stopPropagation();
                    cardWrapper.classList.remove('expanded');
                });
                cardWrapper.appendChild(details);

                container.appendChild(cardWrapper);
            });
        }

        // Check for existing saved report
        const savedReport = Storage.getWeeklyReport(year, month, week);

        // Check for temp save
        const tempData = JSON.parse(localStorage.getItem('TempWeeklyReport') || 'null');
        const isTempValid = tempData &&
            tempData.month == month &&
            tempData.week == week &&
            (Date.now() - tempData.timestamp < 15 * 60 * 1000);

        if (savedReport) {
            document.getElementById('week-title').value = savedReport.title || '';
            document.getElementById('week-details').value = savedReport.details || '';
            document.getElementById('weekly-rating-slider').value = savedReport.rating || 5;
        } else if (isTempValid) {
            document.getElementById('week-title').value = tempData.title || '';
            document.getElementById('week-details').value = tempData.details || '';
            document.getElementById('weekly-rating-slider').value = tempData.rating || 5;
        } else {
            document.getElementById('week-title').value = '';
            document.getElementById('week-details').value = '';

            // Calculate average rating
            if (journals.length > 0) {
                const avg = journals.reduce((s, j) => s + parseFloat(j.dailyRating), 0) / journals.length;
                // Round to nearest 0.25
                const roundedAvg = Math.round(avg * 4) / 4;
                document.getElementById('weekly-rating-slider').value = roundedAvg;
            } else {
                document.getElementById('weekly-rating-slider').value = 5;
            }
        }

        this.updateWeeklyEmoji(parseFloat(document.getElementById('weekly-rating-slider').value));
    },

    /**
     * Toggle day details - inline on desktop, modal on mobile
     */
    toggleDayDetails(wrapper, journal) {
        // Check if mobile (screen width < 768)
        if (window.innerWidth < 768) {
            // Show modal on mobile
            this.showDayModal(journal);
        } else {
            // Toggle inline expansion on desktop
            const wasExpanded = wrapper.classList.contains('expanded');

            // Close all other expanded cards
            document.querySelectorAll('.day-card-wrapper.expanded').forEach(w => {
                w.classList.remove('expanded');
            });

            if (!wasExpanded) {
                wrapper.classList.add('expanded');
                // Scroll to show the expanded content
                wrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    },

    /**
     * Show day details in modal (for mobile)
     */
    showDayModal(journal) {
        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 90%; max-height: 80vh; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border-color); padding-bottom: 10px;">
                    <span>${journal.date}</span>
                    <h3 style="color: var(--secondary-color); margin: 0;">${journal.title}</h3>
                    <span>${journal.emoji || '⭐'} ${journal.dailyRating}/10</span>
                </div>
                <div style="white-space: pre-wrap; text-align: center; line-height: 1.7;">${journal.details}</div>
                <button class="secondary-btn" style="margin-top: 20px; width: 100%;">Close</button>
            </div>
        `;
        overlay.querySelector('button').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });
        document.body.appendChild(overlay);
    },

    /**
     * Show full day content (legacy - redirects to toggle)
     */
    showFullDay(journal) {
        // Use modal for backward compatibility
        this.showDayModal(journal);
    },

    /**
     * Activate panel (write or read)
     */
    activatePanel(panel) {
        document.getElementById('write-panel').classList.toggle('active-focus', panel === 'write');
        document.getElementById('read-panel').classList.toggle('active-focus', panel === 'read');
    },

    /**
     * Update weekly emoji display
     */
    updateWeeklyEmoji(rating) {
        const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
        const idx = Math.min(Math.floor((rating / 9) * 6), 6);
        document.getElementById('weekly-emoji').textContent = emojis[idx];
        document.getElementById('weekly-rating-value').textContent = `${rating.toFixed(2)}/10`;
    },

    /**
     * Save weekly report
     */
    save() {
        const year = parseInt(document.getElementById('report-year')?.value || new Date().getFullYear());
        const month = parseInt(document.getElementById('report-month').value);
        const week = parseInt(document.getElementById('report-week').value);

        if (!month || !week) {
            Toast.warning('Please select month and week!');
            return;
        }

        const report = {
            id: `${year}-${month}-${week}`,
            year,
            month,
            week,
            title: document.getElementById('week-title').value,
            details: document.getElementById('week-details').value,
            rating: document.getElementById('weekly-rating-slider').value,
            savedAt: new Date().toISOString()
        };

        Storage.addWeeklyReport(report);
        localStorage.removeItem('TempWeeklyReport');
        Toast.success('Weekly report saved successfully!');
    },

    /**
     * Show weekly reports list
     */
    showList() {
        this.listPopup.classList.add('active');
        this.renderList();
    },

    /**
     * Hide weekly reports list
     */
    hideList() {
        this.listPopup.classList.remove('active');
    },

    /**
     * Render weekly reports list
     */
    renderList() {
        const grid = document.getElementById('weekly-reports-grid');
        const reports = Storage.getWeeklyReports().sort((a, b) =>
            new Date(b.savedAt) - new Date(a.savedAt)
        );

        grid.innerHTML = '';

        if (reports.length === 0) {
            grid.innerHTML = '<p class="placeholder-msg" style="grid-column: 1/-1;">No saved weekly reports.</p>';
            return;
        }

        reports.forEach((report, index) => {
            const card = document.createElement('div');
            card.className = 'archive-card';
            card.innerHTML = `
                <h4>${report.title}</h4>
                <div class="date-info">Month: ${report.month} | Week: ${report.week}</div>
                <div class="preview">${(report.details || '').substring(0, 80)}...</div>
                <div class="card-btns">
                    <button class="secondary-btn view-btn">View</button>
                    <button class="primary-btn edit-btn">Edit</button>
                    <button class="danger-btn del-btn">Delete</button>
                </div>
            `;

            card.querySelector('.view-btn').addEventListener('click', () => this.viewReport(report));
            card.querySelector('.edit-btn').addEventListener('click', () => this.editReport(report));
            card.querySelector('.del-btn').addEventListener('click', () => this.deleteReport(index));

            grid.appendChild(card);
        });
    },

    /**
     * View a report
     */
    viewReport(report) {
        const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
        const idx = Math.min(Math.floor((parseFloat(report.rating) / 10) * 6), 6);

        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;">
                    <span>Month: ${report.month} | Week: ${report.week}</span>
                    <h2 style="color: var(--secondary-color); margin: 0;">${report.title || 'Weekly Report'}</h2>
                    <span>${emojis[idx]} ${report.rating}/10</span>
                </div>
                <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; text-align: center; direction: rtl; font-size: 1.2rem; line-height: 1.8;">
                    ${report.details || 'No details.'}
                </div>
                
                <!-- Mini Chart -->
                <div style="margin-top: 30px; background: var(--card-bg); padding: 20px; border-radius: 10px; border: 1px solid var(--border-color);">
                    <h3 style="text-align: center; margin-bottom: 15px; color: var(--text-secondary);">📊 Week Statistics</h3>
                    <canvas id="weekly-view-chart" style="max-height: 250px;"></canvas>
                </div>
                
                <button class="secondary-btn" style="margin-top: 25px;">Close</button>
            </div>
        `;

        overlay.querySelector('button').addEventListener('click', () => {
            // Destroy chart before removing
            if (this.viewChart) {
                this.viewChart.destroy();
                this.viewChart = null;
            }
            overlay.remove();
        });

        document.body.appendChild(overlay);

        // Render chart after modal is in DOM
        setTimeout(() => this.renderWeeklyViewChart(report), 50);
    },

    /**
     * Render mini chart for weekly report view
     */
    renderWeeklyViewChart(report) {
        const canvas = document.getElementById('weekly-view-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Get daily journal ratings for this week
        const year = report.year || new Date().getFullYear();
        const month = report.month - 1; // JS months are 0-indexed
        const startDay = (report.week - 1) * 7 + 1;
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        const endDay = Math.min(report.week * 7, 28);

        const labels = [];
        const dailyRatings = [];
        const journals = Storage.getJournals();

        for (let day = startDay; day <= endDay; day++) {
            const date = new Date(year, month, day);
            // Fix: Use local date components
            const y = date.getFullYear();
            const m = String(date.getMonth() + 1).padStart(2, '0');
            const dStr = String(date.getDate()).padStart(2, '0');
            const dateStr = `${y}-${m}-${dStr}`;

            labels.push(`Day ${day}`);

            // Find journal for this date
            const journal = journals.find(j => j.date === dateStr);
            dailyRatings.push(journal ? parseFloat(journal.dailyRating) : null);
        }

        // Calculate weekly average from dailyrating
        const validRatings = dailyRatings.filter(r => r !== null);
        const weeklyAverage = validRatings.length > 0
            ? validRatings.reduce((a, b) => a + b, 0) / validRatings.length
            : 0;

        // Create chart
        if (this.viewChart) this.viewChart.destroy();

        const backgroundColors = [];
        const borderColors = [];
        const processedData = dailyRatings.map(r => {
            if (r === null) {
                backgroundColors.push('rgba(239, 68, 68, 0.6)'); // Red
                borderColors.push('rgba(239, 68, 68, 1)');
                return 0.2; // Small height to make it visible
            }
            backgroundColors.push('rgba(99, 102, 241, 0.6)'); // Indigo
            borderColors.push('rgba(99, 102, 241, 1)');
            return r;
        });

        this.viewChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Daily Rating',
                        data: processedData,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 2,
                        borderRadius: 5
                    },
                    {
                        label: 'Weekly Average',
                        data: Array(labels.length).fill(weeklyAverage),
                        type: 'line',
                        borderColor: 'rgba(34, 197, 94, 1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: 'Weekly Report Rating',
                        data: Array(labels.length).fill(parseFloat(report.rating)),
                        type: 'line',
                        borderColor: 'rgba(251, 146, 60, 1)',
                        borderWidth: 3,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.6, // Taller chart
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10,
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            stepSize: 1, // Show all integers 0-10
                            callback: function (value) {
                                return Math.floor(value) === value ? value : '';
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            font: { size: 10 }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 35
                    }
                }
            },
            plugins: [{
                id: 'emojiLabels',
                afterDatasetsDraw: (chart) => {
                    const ctx = chart.ctx;
                    const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];

                    const meta = chart.getDatasetMeta(0);
                    if (!meta || !meta.data) return;

                    meta.data.forEach((bar, index) => {
                        const rating = dailyRatings[index];
                        if (rating !== null) {
                            const emojiIdx = Math.min(Math.floor((rating / 10) * 6), 6);
                            const emoji = emojis[emojiIdx];

                            // Calculate middle of the bar
                            const barTop = bar.y;
                            const barBottom = bar.base;
                            const middleY = barTop + ((barBottom - barTop) / 2);

                            ctx.save();
                            ctx.font = '24px Arial';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(emoji, bar.x, middleY);
                            ctx.restore();
                        }
                    });
                }
            }]
        });
    },

    /**
     * Edit a report (open in editor)
     */
    editReport(report) {
        this.hideList();
        this.openEditor();
        document.getElementById('report-month').value = report.month;
        document.getElementById('report-week').value = report.week;
        this.loadWeekData();
    },

    /**
     * Delete a report
     */
    deleteReport(index) {
        CustomAlert.confirm('Are you sure you want to delete this weekly report?', (confirmed) => {
            if (confirmed) {
                Storage.deleteWeeklyReport(index);
                this.renderList();
                Toast.success('Report deleted');
            }
        }, 'Delete Report');
    },

    /**
     * Export weekly reports to CSV
     */
    exportCSV() {
        const reports = Storage.getWeeklyReports();
        if (reports.length === 0) {
            Toast.info('No reports to export.');
            return;
        }

        const csv = 'Year,Month,Week,Title,Rating,Details\n' +
            reports.map(r =>
                `${r.year},${r.month},${r.week},"${r.title || ''}",${r.rating},"${(r.details || '').replace(/"/g, '""').replace(/\n/g, ' ')}"`
            ).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'weekly_reports.csv';
        link.click();
        URL.revokeObjectURL(url);
    },

    /**
     * Generate Weekly Report via API
     */
    async generateReport() {
        // Show Busy Alert/GIF
        const overlay = document.createElement('div');
        overlay.className = 'custom-alert-overlay active';
        overlay.id = 'busy-overlay';
        overlay.innerHTML = `
            <div class="custom-alert-box" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 20px;">🤖</div>
                <h3>Rona is busy...</h3>
                <p>Analyzing your week...</p>
                <!-- Placeholder for User's GIF -->
                <img src="assets/loading.gif" alt="Loading..." style="max-width: 100px; margin: 20px auto; display: block;">
            </div>
        `;
        document.body.appendChild(overlay);
        
        try {
            const response = await fetch('/api/weekly_report', { method: 'POST' });
            const result = await response.json();
            
            document.body.removeChild(overlay);
            
            if (result.ok) {
                // Open Editor and Fill Data
                this.openEditor();
                document.getElementById('week-title').value = `Weekly Report - ${new Date().toLocaleDateString()}`;
                
                // Set the generated content
                const detailsBox = document.getElementById('week-details');
                detailsBox.value = result.report;
                
                // Read-only mode as requested
                detailsBox.setAttribute('readonly', true);
                document.getElementById('week-title').setAttribute('readonly', true);
                
                Toast.success('Weekly Report Generated!');
            } else {
                Toast.error('Failed to generate report: ' + result.error);
            }
        } catch (error) {
            if (document.getElementById('busy-overlay')) document.body.removeChild(overlay);
            Toast.error('Network error generating report');
            console.error(error);
        }
    }
};

window.Weekly = Weekly;
