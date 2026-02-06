/**
 * Monthly Reports Module
 */

const Monthly = {
    popup: null,
    listPopup: null,
    viewMode: 'weeks', // 'days' or 'weeks'

    /**
     * Initialize monthly module
     */
    init() {
        this.popup = document.getElementById('monthly-popup');
        this.listPopup = document.getElementById('monthly-list-popup');

        if (this.popup) {
            this.setupListeners();
        }
    },

    /**
     * Setup event listeners
     */
    setupListeners() {
        // Open/close monthly popup
        document.getElementById('monthly-report-btn')?.addEventListener('click', () => this.openEditor());
        document.getElementById('close-monthly-popup')?.addEventListener('click', () => this.closeEditor());

        // Month selection
        document.getElementById('monthly-report-month')?.addEventListener('change', () => this.loadMonthData());

        // Save monthly report
        document.getElementById('save-monthly-btn')?.addEventListener('click', () => this.save());

        // History buttons
        document.getElementById('monthly-history-btn')?.addEventListener('click', () => this.showList());
        document.getElementById('show-monthly-list-btn')?.addEventListener('click', () => this.showList());
        document.getElementById('close-monthly-list')?.addEventListener('click', () => this.hideList());
        document.getElementById('export-monthly-btn')?.addEventListener('click', () => this.exportCSV());

        // View mode toggle
        document.getElementById('view-mode-toggle')?.addEventListener('click', () => this.toggleViewMode());

        // Monthly rating slider
        document.getElementById('monthly-rating-slider')?.addEventListener('input', (e) => {
            this.updateMonthlyEmoji(parseFloat(e.target.value));
        });

        // Year selector change
        document.getElementById('monthly-report-year')?.addEventListener('change', () => this.loadMonthData());

        // Create New button in history
        document.getElementById('create-new-monthly-btn')?.addEventListener('click', () => {
            this.hideList();
            this.openEditor();
        });

        // Panel focus activation (like weekly)
        const monthlyWritePanel = this.popup?.querySelector('.write-panel');
        const monthlyReadPanel = this.popup?.querySelector('.read-panel');

        if (monthlyWritePanel) {
            monthlyWritePanel.addEventListener('click', () => this.activatePanel('write'));
        }
        if (monthlyReadPanel) {
            monthlyReadPanel.addEventListener('click', () => this.activatePanel('read'));
        }

        // Populate years
        this.populateYears();
    },

    /**
     * Populate year dropdown (current year ±5)
     */
    populateYears() {
        const yearSelect = document.getElementById('monthly-report-year');
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
     * Activate panel (write or read) - focus expansion
     */
    activatePanel(panel) {
        const writePanel = this.popup?.querySelector('.write-panel');
        const readPanel = this.popup?.querySelector('.read-panel');
        writePanel?.classList.toggle('active-focus', panel === 'write');
        readPanel?.classList.toggle('active-focus', panel === 'read');
    },

    /**
     * Toggle between days and weeks view
     */
    toggleViewMode() {
        this.viewMode = this.viewMode === 'weeks' ? 'days' : 'weeks';
        const btn = document.getElementById('view-mode-toggle');
        btn.textContent = this.viewMode === 'weeks' ? '📅 Show Days' : '📆 Show Weeks';
        this.loadMonthData();
    },

    /**
     * Open monthly editor
     */
    openEditor() {
        const now = new Date();
        const currentMonth = now.getMonth() + 1;

        document.getElementById('monthly-report-month').value = currentMonth;
        this.popup.classList.add('open');
        this.loadMonthData();
    },

    /**
     * Close monthly editor
     */
    closeEditor() {
        this.popup.classList.remove('open');
    },

    /**
     * Load month data
     */
    loadMonthData() {
        const year = parseInt(document.getElementById('monthly-report-year')?.value || new Date().getFullYear());
        const month = parseInt(document.getElementById('monthly-report-month').value);
        if (!month) return;
        const container = document.getElementById('monthly-reference-list');
        container.innerHTML = '';

        if (this.viewMode === 'days') {
            this.loadDaysView(container, year, month);
        } else {
            this.loadWeeksView(container, year, month);
        }

        // Load existing report data
        const savedReport = this.getMonthlyReport(year, month);
        if (savedReport) {
            document.getElementById('monthly-title').value = savedReport.title || '';
            document.getElementById('monthly-details').value = savedReport.details || '';
            document.getElementById('monthly-rating-slider').value = savedReport.rating || 5;
        } else {
            document.getElementById('monthly-title').value = '';
            document.getElementById('monthly-details').value = '';
            document.getElementById('monthly-rating-slider').value = 5;

            // Calculate average rating from daily journals if available
            const journals = Storage.getJournals().filter(j => {
                const d = new Date(j.date);
                return d.getFullYear() === year && (d.getMonth() + 1) === month;
            });

            if (journals.length > 0) {
                const avg = journals.reduce((s, j) => s + parseFloat(j.dailyRating), 0) / journals.length;
                // Round to nearest 0.25
                const roundedAvg = Math.round(avg * 4) / 4;
                document.getElementById('monthly-rating-slider').value = roundedAvg;
            }
        }

        this.updateMonthlyEmoji(parseFloat(document.getElementById('monthly-rating-slider').value));
    },

    /**
     * Load days view - shows all daily journals for the month
     */
    loadDaysView(container, year, month) {
        const journals = Storage.getJournals().filter(j => {
            const d = new Date(j.date);
            return d.getFullYear() === year && (d.getMonth() + 1) === month && d.getDate() <= 28;
        }).sort((a, b) => new Date(a.date) - new Date(b.date));

        if (journals.length === 0) {
            container.innerHTML = '<p class="placeholder-msg">No daily journals for this month.</p>';
            return;
        }

        // Calculate average
        const avg = journals.reduce((s, j) => s + parseFloat(j.dailyRating), 0) / journals.length;
        container.innerHTML = `<div class="month-stats">📊 ${journals.length} days | Avg: ${avg.toFixed(1)}/10</div>`;

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
    },

    /**
     * Toggle day details - inline on desktop, modal on mobile
     */
    toggleDayDetails(wrapper, journal) {
        if (window.innerWidth < 768) {
            this.showFullDay(journal);
        } else {
            const wasExpanded = wrapper.classList.contains('expanded');
            document.querySelectorAll('.day-card-wrapper.expanded').forEach(w => {
                w.classList.remove('expanded');
            });
            if (!wasExpanded) {
                wrapper.classList.add('expanded');
                wrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    },

    /**
     * Load weeks view - shows weekly reports for the month
     */
    loadWeeksView(container, year, month) {
        const weeklyReports = Storage.getWeeklyReports().filter(r =>
            r.year === year && r.month === month
        ).sort((a, b) => a.week - b.week);

        if (weeklyReports.length === 0) {
            container.innerHTML = '<p class="placeholder-msg">No weekly reports for this month.</p>';
            return;
        }

        // Calculate average
        const avg = weeklyReports.reduce((s, r) => s + parseFloat(r.rating), 0) / weeklyReports.length;
        container.innerHTML = `<div class="month-stats">📊 ${weeklyReports.length} weeks | Avg: ${avg.toFixed(1)}/10</div>`;

        weeklyReports.forEach((r, index) => {
            const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
            const idx = Math.min(Math.floor((parseFloat(r.rating) / 9) * 6), 6);

            const cardWrapper = document.createElement('div');
            cardWrapper.className = 'day-card-wrapper';
            cardWrapper.dataset.index = index;

            const card = document.createElement('div');
            card.className = 'mini-week-card';
            card.innerHTML = `
                <div class="mini-card-header">
                    <span class="mini-date">Week ${r.week}</span>
                    <span class="mini-rating">${emojis[idx]} ${r.rating}/10</span>
                </div>
                <div class="mini-title">${r.title}</div>
            `;
            card.addEventListener('click', () => this.toggleWeekDetails(cardWrapper, r));
            cardWrapper.appendChild(card);

            // Create inline details panel (hidden by default)
            const details = document.createElement('div');
            details.className = 'inline-day-details';
            details.innerHTML = `
                <div class="details-content">${r.details || 'No details written.'}</div>
                <button class="collapse-btn">▲ Close</button>
            `;
            details.querySelector('.collapse-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                cardWrapper.classList.remove('expanded');
            });
            cardWrapper.appendChild(details);

            container.appendChild(cardWrapper);
        });
    },

    /**
     * Toggle week details - inline on desktop, modal on mobile
     */
    toggleWeekDetails(wrapper, report) {
        if (window.innerWidth < 768) {
            this.showFullWeek(report);
        } else {
            const wasExpanded = wrapper.classList.contains('expanded');
            document.querySelectorAll('.day-card-wrapper.expanded').forEach(w => {
                w.classList.remove('expanded');
            });
            if (!wasExpanded) {
                wrapper.classList.add('expanded');
                wrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    },

    /**
     * Show full day details
     */
    showFullDay(journal) {
        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 700px; max-height: 80vh; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border-color); padding-bottom: 10px;">
                    <span>${journal.date}</span>
                    <h3 style="color: var(--secondary-color); margin: 0;">${journal.title}</h3>
                    <span>${journal.emoji} ${journal.dailyRating}/10</span>
                </div>
                <div style="white-space: pre-wrap; text-align: center; line-height: 1.7;">${journal.details}</div>
                <button class="secondary-btn" style="margin-top: 20px;">Close</button>
            </div>
        `;
        overlay.querySelector('button').addEventListener('click', () => overlay.remove());
        document.body.appendChild(overlay);
    },

    /**
     * Show full week details
     */
    showFullWeek(report) {
        const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
        const idx = Math.min(Math.floor((parseFloat(report.rating) / 10) * 6), 6);

        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 700px; max-height: 80vh; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid var(--border-color); padding-bottom: 10px;">
                    <span>Week ${report.week}</span>
                    <h3 style="color: var(--secondary-color); margin: 0;">${report.title || 'Weekly Report'}</h3>
                    <span>${emojis[idx]} ${report.rating}/10</span>
                </div>
                <div style="white-space: pre-wrap; text-align: center; line-height: 1.7;">${report.details || 'No details.'}</div>
                <button class="secondary-btn" style="margin-top: 20px;">Close</button>
            </div>
        `;
        overlay.querySelector('button').addEventListener('click', () => overlay.remove());
        document.body.appendChild(overlay);
    },

    /**
     * Update monthly emoji display
     */
    updateMonthlyEmoji(rating) {
        const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
        const idx = Math.min(Math.floor((rating / 10) * 6), 6);
        document.getElementById('monthly-emoji').textContent = emojis[idx];
        document.getElementById('monthly-rating-value').textContent = `${rating.toFixed(2)}/10`;
    },

    /**
     * Save monthly report
     */
    save() {
        const year = parseInt(document.getElementById('monthly-report-year')?.value || new Date().getFullYear());
        const month = parseInt(document.getElementById('monthly-report-month').value);
        if (!month) {
            Toast.warning('Please select a month!');
            return;
        }

        const report = {
            id: `${year}-${month}`,
            year,
            month,
            title: document.getElementById('monthly-title').value,
            details: document.getElementById('monthly-details').value,
            rating: document.getElementById('monthly-rating-slider').value,
            savedAt: new Date().toISOString()
        };

        this.addMonthlyReport(report);
        Toast.success('Monthly report saved successfully!');
    },

    /**
     * Get monthly report from storage
     */
    getMonthlyReport(year, month) {
        return Storage.getMonthlyReport(year, month);
    },

    /**
     * Add/update monthly report
     */
    addMonthlyReport(report) {
        Storage.addMonthlyReport(report);
    },

    /**
     * Get all monthly reports
     */
    getMonthlyReports() {
        return Storage.getMonthlyReports();
    },

    /**
     * Show monthly reports list
     */
    showList() {
        this.listPopup.classList.add('active');
        this.renderList();
    },

    /**
     * Hide monthly reports list
     */
    hideList() {
        this.listPopup.classList.remove('active');
    },

    /**
     * Render monthly reports list
     */
    renderList() {
        const grid = document.getElementById('monthly-reports-grid');
        const reports = this.getMonthlyReports().sort((a, b) =>
            new Date(b.savedAt) - new Date(a.savedAt)
        );

        grid.innerHTML = '';

        if (reports.length === 0) {
            grid.innerHTML = '<p class="placeholder-msg" style="grid-column: 1/-1;">No saved monthly reports.</p>';
            return;
        }

        const monthNames = ['', 'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'];

        reports.forEach((report, index) => {
            const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
            const idx = Math.min(Math.floor((parseFloat(report.rating) / 10) * 6), 6);

            const card = document.createElement('div');
            card.className = 'archive-card';
            card.innerHTML = `
                <h4>${report.title}</h4>
                <div class="date-info">${monthNames[report.month]} ${report.year}</div>
                <div class="card-rating">${emojis[idx]} ${report.rating}/10</div>
                <div class="preview">${(report.details || '').substring(0, 60)}...</div>
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
        const monthNames = ['', 'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'];
        const emojis = ['😩', '😔', '😕', '😐', '😊', '😃', '🤩'];
        const idx = Math.min(Math.floor((parseFloat(report.rating) / 10) * 6), 6);

        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border-color); padding-bottom: 15px;">
                    <span>${monthNames[report.month]} ${report.year}</span>
                    <h2 style="color: var(--secondary-color); margin: 0;">${report.title || 'Monthly Report'}</h2>
                    <span>${emojis[idx]} ${report.rating}/10</span>
                </div>
                <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; text-align: center; line-height: 1.8; margin-bottom: 30px;">
                    ${report.details || 'No details.'}
                </div>
                
                <!-- Mini Chart -->
                <div style="background: var(--card-bg); padding: 20px; border-radius: 10px; border: 1px solid var(--border-color);">
                    <h3 style="text-align: center; margin-bottom: 15px; color: var(--text-secondary);">📊 Month Statistics</h3>
                    <canvas id="monthly-view-chart" style="max-height: 300px;"></canvas>
                </div>
                
                <button class="secondary-btn" style="margin-top: 25px;">Close</button>
            </div>
        `;

        overlay.querySelector('button').addEventListener('click', () => {
            if (this.viewChart) {
                this.viewChart.destroy();
                this.viewChart = null;
            }
            overlay.remove();
        });

        document.body.appendChild(overlay);

        // Render chart after modal is in DOM
        setTimeout(() => this.renderMonthlyViewChart(report), 50);
    },

    /**
     * Render Wavy Bar Chart for Monthly Report
     */
    renderMonthlyViewChart(report) {
        const canvas = document.getElementById('monthly-view-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const journals = Storage.getJournals();
        const weeklyReports = Storage.getWeeklyReports();

        // Prepare 4 weeks data
        const year = report.year;
        const month = report.month - 1; // 0-indexed

        const weeksData = [];
        let totalDailySum = 0;
        let totalDailyCount = 0;
        let totalWeeklySum = 0;
        let totalWeeklyCount = 0;

        for (let w = 1; w <= 4; w++) {
            const startDay = (w - 1) * 7 + 1;
            const endDay = w * 7;

            // 1. Calculate Daily Average (Journals)
            const dailyValues = [];
            const rawValues = []; // Store raw to detect nulls for gradient

            for (let d = startDay; d <= endDay; d++) {
                const date = new Date(year, month, d);
                const y = date.getFullYear();
                const m = String(date.getMonth() + 1).padStart(2, '0');
                const dStr = String(date.getDate()).padStart(2, '0');
                const dateStr = `${y}-${m}-${dStr}`;

                const journal = journals.find(j => j.date === dateStr);
                const rating = journal ? parseFloat(journal.dailyRating) : null;
                dailyValues.push(rating);
                rawValues.push(rating);

                if (rating !== null) {
                    totalDailySum += rating;
                    totalDailyCount++;
                }
            }

            const valid = dailyValues.filter(v => v !== null);
            const avg = valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;

            // Normalize daily variations
            const wavePoints = dailyValues.map(v => (v !== null ? v : avg));

            weeksData.push({
                weekIndex: w,
                average: avg,
                dailyPoints: wavePoints,
                rawValues: rawValues
            });

            // 2. Get Weekly Report Rating
            const weeklyRep = weeklyReports.find(r =>
                r.year == year &&
                r.month == (month + 1) &&
                r.week == w
            );
            if (weeklyRep) {
                totalWeeklySum += parseFloat(weeklyRep.rating);
                totalWeeklyCount++;
            }
        }

        // Global Averages
        const globalDailyAvg = totalDailyCount > 0 ? totalDailySum / totalDailyCount : 0;
        const globalWeeklyAvg = totalWeeklyCount > 0 ? totalWeeklySum / totalWeeklyCount : 0;

        const labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];

        // Create chart
        if (this.viewChart) this.viewChart.destroy();

        // Custom Plugin for Wavy Bars
        const wavyBarPlugin = {
            id: 'wavyBars',
            afterDatasetsDraw: (chart) => {
                const ctx = chart.ctx;
                const meta = chart.getDatasetMeta(0); // The dataset for wavy bars

                meta.data.forEach((bar, index) => {
                    const weekData = weeksData[index];
                    if (!weekData || weekData.average === 0) return;

                    const x = bar.x;
                    const base = bar.base;
                    const width = bar.width;
                    const scale = chart.scales.y;

                    const left = x - width / 2;
                    const right = x + width / 2;

                    // Create Gradient for Interruption Detection
                    const gradient = ctx.createLinearGradient(left, 0, right, 0);
                    const rawValues = weekData.rawValues;
                    const totalDays = rawValues.length;

                    // Map days to gradient stops
                    // If day is null -> Red, else -> Blue
                    // Sharp transitions? Or smooth? Sharp looks better for discrete days.
                    // Stop spacing: 0 to 1/7, 1/7 to 2/7...

                    const dayWidth = 1 / totalDays;

                    rawValues.forEach((val, i) => {
                        const color = (val === null) ? 'rgba(239, 68, 68, 0.6)' : 'rgba(99, 102, 241, 0.6)'; // Red vs Indigo
                        const start = i * dayWidth;
                        const end = (i + 1) * dayWidth;

                        // Use two stops for solid block of color per day
                        gradient.addColorStop(start, color);
                        gradient.addColorStop(end, color);
                    });

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(left, base);

                    const dailyPoints = weekData.dailyPoints;
                    const stepX = width / (dailyPoints.length - 1 || 1);

                    const firstVal = dailyPoints[0];
                    const firstDiff = Math.max(-0.5, Math.min(0.5, (firstVal - weekData.average) * 0.1));
                    const firstY = scale.getPixelForValue(weekData.average + firstDiff);

                    ctx.lineTo(left, firstY);

                    for (let i = 0; i < dailyPoints.length - 1; i++) {
                        const currX = left + (i * stepX);
                        const nextX = left + ((i + 1) * stepX);

                        const currDiff = Math.max(-0.5, Math.min(0.5, (dailyPoints[i] - weekData.average) * 0.15));
                        const nextDiff = Math.max(-0.5, Math.min(0.5, (dailyPoints[i + 1] - weekData.average) * 0.15));

                        const currY = scale.getPixelForValue(weekData.average + currDiff);
                        const nextY = scale.getPixelForValue(weekData.average + nextDiff);

                        const midX = (currX + nextX) / 2;
                        const midY = (currY + nextY) / 2;

                        ctx.quadraticCurveTo(currX, currY, midX, midY);
                    }

                    const lastVal = dailyPoints[dailyPoints.length - 1];
                    const lastDiff = Math.max(-0.5, Math.min(0.5, (lastVal - weekData.average) * 0.15));
                    const lastY = scale.getPixelForValue(weekData.average + lastDiff);
                    ctx.lineTo(right, lastY);

                    ctx.lineTo(right, base);
                    ctx.closePath();

                    ctx.fillStyle = gradient; // Use the gradient!
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)'; // Lighter border, or maybe split border?
                    // Gradient border is hard. Let's keep a subtle border to define shape
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                });
            }
        };

        this.viewChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Week Performance',
                        data: [0, 0, 0, 0],
                        backgroundColor: 'rgba(99, 102, 241, 0.6)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1,
                    },
                    {
                        label: 'Avg of Days',
                        data: Array(4).fill(globalDailyAvg),
                        type: 'line',
                        borderColor: 'rgba(14, 165, 233, 0.8)', // Sky Blue
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0
                    },
                    {
                        label: 'Avg of Weekly Reports',
                        data: Array(4).fill(globalWeeklyAvg),
                        type: 'line',
                        borderColor: 'rgba(34, 197, 94, 0.8)', // Green
                        borderWidth: 2,
                        borderDash: [2, 2],
                        pointRadius: 0
                    },
                    {
                        label: 'Monthly Rating',
                        data: Array(4).fill(parseFloat(report.rating)),
                        type: 'line',
                        borderColor: 'rgba(251, 146, 60, 1)', // Orange
                        borderWidth: 3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.6,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10,
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            stepSize: 1
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#fff',
                            font: { size: 11 },
                            generateLabels: (chart) => {
                                // Default labels
                                const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                // Add "Interruption" label manually
                                labels.push({
                                    text: 'Interruption (Red)',
                                    fillStyle: 'rgba(239, 68, 68, 0.6)',
                                    strokeStyle: 'rgba(239, 68, 68, 1)',
                                    lineWidth: 1,
                                    hidden: false,
                                    index: 5 // Force to end
                                });
                                return labels;
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                // Custom tooltip for wavy bars?
                                // Maybe just show "Week X" details.
                                // Ideally, if hovering red part -> "Interruption".
                                // That requires mouse x position mapping...
                                // For now, let's just stick to default values or Avg.
                                return context.formattedValue;
                            }
                        }
                    }
                }
            },
            plugins: [wavyBarPlugin]
        });
    },

    /**
     * Edit a report
     */
    editReport(report) {
        this.hideList();
        this.openEditor();
        document.getElementById('monthly-report-month').value = report.month;
        this.loadMonthData();
    },

    /**
     * Delete a report
     */
    deleteReport(index) {
        CustomAlert.confirm('Are you sure you want to delete this monthly report?', (confirmed) => {
            if (confirmed) {
                Storage.deleteMonthlyReport(index);
                this.renderList();
                Toast.success('Report deleted');
            }
        }, 'Delete Report');
    },

    /**
     * Export monthly reports to CSV
     */
    exportCSV() {
        const reports = this.getMonthlyReports();
        if (reports.length === 0) {
            Toast.info('No reports to export.');
            return;
        }

        const csv = 'Year,Month,Title,Rating,Details\n' +
            reports.map(r =>
                `${r.year},${r.month},"${r.title || ''}",${r.rating},"${(r.details || '').replace(/"/g, '""').replace(/\n/g, ' ')}"`
            ).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'monthly_reports.csv';
        link.click();
        URL.revokeObjectURL(url);
    }
};

window.Monthly = Monthly;
