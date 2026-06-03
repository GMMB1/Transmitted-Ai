/**
 * Statistics Module - Enhanced Chart.js with scroll and multiple data types
 */

const Statistics = {
    chart: null,
    page: null,
    currentMode: 'journals', // journals, weekly, monthly, both
    currentGrouping: 'daily', // daily, weekly, monthly
    viewWindow: { start: 0, maxVisible: 30 }, // Max 30 points visible at once
    allData: { labels: [], journals: [], weekly: [], monthly: [] },

    /**
     * Initialize statistics module
     */
    init() {
        this.page = document.getElementById('statistics-page');
        this.setupEventListeners();
    },

    /**
     * Set default date range to last continuous period
     */
    setDefaultRange() {
        const journals = Storage.getJournals();

        if (journals.length === 0) {
            // No data - set to current month
            const now = new Date();
            const firstDay = new Date(now.getFullYear(), now.getMonth(), 1).toISOString().split('T')[0];
            const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0).toISOString().split('T')[0];
            document.getElementById('stats-start-date').value = firstDay;
            document.getElementById('stats-end-date').value = lastDay;
            return;
        }

        // Sort journals by date (newest first)
        const sortedJournals = [...journals].sort((a, b) => new Date(b.date) - new Date(a.date));

        // Find last continuous period (from latest back to first gap)
        const latestDate = new Date(sortedJournals[0].date);
        let startDate = latestDate;

        for (let i = 1; i < sortedJournals.length; i++) {
            const currentDate = new Date(sortedJournals[i].date);
            const previousDate = new Date(sortedJournals[i - 1].date);

            // Calculate gap in days
            const gapDays = Math.floor((previousDate - currentDate) / (1000 * 60 * 60 * 24));

            if (gapDays > 1) {
                // Found a gap - stop here
                break;
            }

            // Continue - this is part of continuous period
            startDate = currentDate;
        }

        // Set the date inputs
        document.getElementById('stats-start-date').value = startDate.toISOString().split('T')[0];
        document.getElementById('stats-end-date').value = latestDate.toISOString().split('T')[0];
    },

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.getElementById('show-stats-btn').addEventListener('click', () => this.show());
        document.getElementById('back-to-home').addEventListener('click', () => this.hide());

        // Stats mode selector
        const modeSelector = document.getElementById('stats-mode');
        if (modeSelector) {
            modeSelector.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.show();
            });
        }

        // Stats grouping selector
        const groupingSelector = document.getElementById('stats-grouping');
        if (groupingSelector) {
            groupingSelector.addEventListener('change', (e) => {
                this.currentGrouping = e.target.value;
                this.show();
            });
        }

        // Quick range selector
        const quickRangeSelector = document.getElementById('quick-range-selector');
        if (quickRangeSelector) {
            quickRangeSelector.addEventListener('change', (e) => {
                this.applyQuickRange(e.target.value);
            });
        }

        // Manual date change - switch to custom
        document.getElementById('stats-start-date')?.addEventListener('change', () => {
            if (quickRangeSelector) quickRangeSelector.value = 'custom';
        });
        document.getElementById('stats-end-date')?.addEventListener('change', () => {
            if (quickRangeSelector) quickRangeSelector.value = 'custom';
        });

        // View window selector
        document.getElementById('view-window-selector')?.addEventListener('change', (e) => {
            const value = e.target.value;
            this.viewWindow.maxVisible = value === 'all' ? 999999 : parseInt(value);
            // Re-render if data already loaded
            if (this.allData.labels.length > 0) {
                this.viewWindow.start = Math.max(0, this.allData.labels.length - this.viewWindow.maxVisible);
                this.renderCurrentView();
                this.updateScrollInfo();
                this.toggleScrollButtons();
            }
        });

        // Chart scroll buttons
        document.getElementById('scroll-left')?.addEventListener('click', () => this.scrollChart(-10));
        document.getElementById('scroll-right')?.addEventListener('click', () => this.scrollChart(10));
    },

    /**
     * Apply quick range selection
     */
    applyQuickRange(range) {
        if (range === 'custom') return; // Don't change dates for custom

        const now = new Date();
        let startDate, endDate;

        switch (range) {
            case 'week':
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                endDate = now;
                break;
            case '2weeks':
                startDate = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);
                endDate = now;
                break;
            case 'month':
                startDate = new Date(now.getFullYear(), now.getMonth(), 1);
                endDate = new Date(now.getFullYear(), now.getMonth() + 1, 0);
                break;
            case '3months':
                startDate = new Date(now.getFullYear(), now.getMonth() - 2, 1);
                endDate = now;
                break;
            case 'year':
                startDate = new Date(now.getFullYear(), 0, 1);
                endDate = new Date(now.getFullYear(), 11, 31);
                break;
        }

        if (startDate && endDate) {
            document.getElementById('stats-start-date').value = startDate.toISOString().split('T')[0];
            document.getElementById('stats-end-date').value = endDate.toISOString().split('T')[0];
            // User clicks "Show" button manually
        }
    },

    /**
     * Show statistics page
     */
    showPage() {
        this.page.classList.remove('hidden');
        // Recalculate date range each time page is shown
        this.setDefaultRange();
    },

    /**
     * Hide statistics page and return to home
     */
    hide() {
        this.page.classList.add('hidden');
        // Show main container
        document.querySelector('.container').style.display = 'block';
        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.page === 'home');
        });
    },

    /**
     * Scroll chart left/right
     */
    scrollChart(delta) {
        const maxStart = Math.max(0, this.allData.labels.length - this.viewWindow.maxVisible);
        this.viewWindow.start = Math.max(0, Math.min(maxStart, this.viewWindow.start + delta));
        this.renderCurrentView();
        this.updateScrollInfo();
    },

    /**
     * Update scroll info display
     */
    updateScrollInfo() {
        const infoEl = document.getElementById('scroll-info');
        if (infoEl) {
            const end = Math.min(this.viewWindow.start + this.viewWindow.maxVisible, this.allData.labels.length);
            infoEl.textContent = `Showing ${this.viewWindow.start + 1} - ${end} of ${this.allData.labels.length} days`;
        }
    },

    /**
     * Show statistics - main function
     */
    show() {
        const startDate = document.getElementById('stats-start-date').value;
        const endDate = document.getElementById('stats-end-date').value;

        if (!startDate || !endDate) {
            Toast.warning('Please select start and end dates.');
            return;
        }

        const start = new Date(startDate);
        const end = new Date(endDate);

        // Generate all dates in range (for gaps)
        this.generateDateRange(start, end);

        // Get data based on mode
        this.populateData();

        // Apply grouping BEFORE rendering
        if (this.currentGrouping !== 'daily') {
            this.applyGrouping();
        }

        // Reset scroll position to show latest data
        this.viewWindow.start = Math.max(0, this.allData.labels.length - this.viewWindow.maxVisible);

        // Render with scroll
        this.renderCurrentView();
        this.updateScrollInfo();
        this.toggleScrollButtons();
    },

    /**
     * Generate all dates in range (including empty days)
     */
    generateDateRange(start, end) {
        this.allData.labels = [];
        this.allData.journals = [];
        this.allData.weekly = [];
        this.allData.monthly = [];

        const current = new Date(start);
        while (current <= end) {
            // Skip days 29, 30, 31
            if (current.getDate() <= 28) {
                this.allData.labels.push(current.toISOString().split('T')[0]);
            }
            current.setDate(current.getDate() + 1);
        }
    },

    /**
     * Populate data arrays with actual values (null for gaps)
     */
    populateData() {
        const journals = Storage.getJournals();
        const weeklyReports = Storage.getWeeklyReports();
        const monthlyReports = Storage.getMonthlyReports();

        // Create lookup maps
        const journalMap = {};
        journals.forEach(j => {
            journalMap[j.date] = parseFloat(j.dailyRating);
        });

        // Weekly reports - map to their date ranges
        const weeklyMap = {};
        weeklyReports.forEach(r => {
            const year = r.year || new Date().getFullYear();
            const month = r.month - 1;
            const startDay = (r.week - 1) * 7 + 1;
            const endDay = Math.min(r.week * 7, 28);

            for (let day = startDay; day <= endDay; day++) {
                const d = new Date(year, month, day);
                const dateStr = d.toISOString().split('T')[0];
                weeklyMap[dateStr] = parseFloat(r.rating);
            }
        });

        // Monthly reports - map to entire month
        const monthlyMap = {};
        monthlyReports.forEach(r => {
            const year = r.year || new Date().getFullYear();
            const month = r.month - 1;
            const daysInMonth = 28; // Strict 28-day month for stats

            for (let day = 1; day <= daysInMonth; day++) {
                const d = new Date(year, month, day);
                const dateStr = d.toISOString().split('T')[0];
                monthlyMap[dateStr] = parseFloat(r.rating);
            }
        });

        // Fill data arrays
        this.allData.labels.forEach(date => {
            this.allData.journals.push(journalMap[date] !== undefined ? journalMap[date] : null);
            this.allData.weekly.push(weeklyMap[date] !== undefined ? weeklyMap[date] : null);
            this.allData.monthly.push(monthlyMap[date] !== undefined ? monthlyMap[date] : null);
        });
    },

    /**
     * Apply grouping (weekly or monthly averages)
     */
    applyGrouping() {
        if (this.currentGrouping === 'daily') return; // No grouping needed

        const grouped = {
            labels: [],
            journals: [],
            weekly: [],
            monthly: []
        };

        if (this.currentGrouping === 'weekly') {
            // Group by week (7 days)
            for (let i = 0; i < this.allData.labels.length; i += 7) {
                const weekLabels = this.allData.labels.slice(i, i + 7);
                const weekJournals = this.allData.journals.slice(i, i + 7).filter(v => v !== null);
                const weekWeekly = this.allData.weekly.slice(i, i + 7).filter(v => v !== null);
                const weekMonthly = this.allData.monthly.slice(i, i + 7).filter(v => v !== null);

                // Calculate averages
                const avgJournals = weekJournals.length > 0 ? weekJournals.reduce((a, b) => a + b, 0) / weekJournals.length : null;
                const avgWeekly = weekWeekly.length > 0 ? weekWeekly.reduce((a, b) => a + b, 0) / weekWeekly.length : null;
                const avgMonthly = weekMonthly.length > 0 ? weekMonthly.reduce((a, b) => a + b, 0) / weekMonthly.length : null;

                // Label: Week of start date
                const startDate = new Date(weekLabels[0]);
                const endDate = new Date(weekLabels[weekLabels.length - 1]);
                grouped.labels.push(`Week ${startDate.getDate()}/${startDate.getMonth() + 1}`);
                grouped.journals.push(avgJournals);
                grouped.weekly.push(avgWeekly);
                grouped.monthly.push(avgMonthly);
            }
        } else if (this.currentGrouping === 'monthly') {
            // Group by month
            const monthsMap = {};

            this.allData.labels.forEach((date, idx) => {
                const d = new Date(date);
                const monthKey = `${d.getFullYear()}-${d.getMonth() + 1}`;

                if (!monthsMap[monthKey]) {
                    monthsMap[monthKey] = {
                        label: `${d.toLocaleString('default', { month: 'short' })} ${d.getFullYear()}`,
                        journals: [],
                        weekly: [],
                        monthly: []
                    };
                }

                if (this.allData.journals[idx] !== null) monthsMap[monthKey].journals.push(this.allData.journals[idx]);
                if (this.allData.weekly[idx] !== null) monthsMap[monthKey].weekly.push(this.allData.weekly[idx]);
                if (this.allData.monthly[idx] !== null) monthsMap[monthKey].monthly.push(this.allData.monthly[idx]);
            });

            // Convert to arrays
            Object.keys(monthsMap).sort().forEach(key => {
                const month = monthsMap[key];
                grouped.labels.push(month.label);
                grouped.journals.push(month.journals.length > 0 ? month.journals.reduce((a, b) => a + b, 0) / month.journals.length : null);
                grouped.weekly.push(month.weekly.length > 0 ? month.weekly.reduce((a, b) => a + b, 0) / month.weekly.length : null);
                grouped.monthly.push(month.monthly.length > 0 ? month.monthly.reduce((a, b) => a + b, 0) / month.monthly.length : null);
            });
        }

        // Replace allData with grouped data
        this.allData = grouped;
    },

    /**
     * Toggle scroll buttons visibility
     */
    toggleScrollButtons() {
        const scrollControls = document.getElementById('scroll-controls');
        if (scrollControls) {
            scrollControls.style.display = this.allData.labels.length > this.viewWindow.maxVisible ? 'flex' : 'none';
        }
    },

    /**
     * Render current view of the chart
     */
    renderCurrentView() {
        const ctx = document.getElementById('stats-chart').getContext('2d');

        if (this.chart) {
            this.chart.destroy();
        }

        const start = this.viewWindow.start;
        const end = Math.min(start + this.viewWindow.maxVisible, this.allData.labels.length);

        const labels = this.allData.labels.slice(start, end);
        const journalData = this.allData.journals.slice(start, end);
        const weeklyData = this.allData.weekly.slice(start, end);
        const monthlyData = this.allData.monthly.slice(start, end);

        // Colors
        const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color').trim();
        const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--secondary-color').trim();
        const accentColor = '#22c55e'; // Green for monthly

        // Build datasets based on mode
        const datasets = [];

        if (this.currentMode === 'journals' || this.currentMode === 'both') {
            const validJournals = journalData.filter(v => v !== null);
            const journalAvg = validJournals.length > 0
                ? validJournals.reduce((a, b) => a + b, 0) / validJournals.length
                : 0;

            datasets.push({
                label: `Daily Journals (Avg: ${journalAvg.toFixed(1)})`,
                data: journalData,
                borderColor: primaryColor,
                backgroundColor: primaryColor + '30',
                fill: false,
                tension: 0.4,
                spanGaps: false,
                pointRadius: 5,
                pointHoverRadius: 8,
                pointBackgroundColor: journalData.map(v => v !== null ? primaryColor : 'transparent'),
                borderWidth: 3
            });
        }

        if (this.currentMode === 'weekly' || this.currentMode === 'both') {
            const validWeekly = weeklyData.filter(v => v !== null);
            const weeklyAvg = validWeekly.length > 0
                ? validWeekly.reduce((a, b) => a + b, 0) / validWeekly.length
                : 0;

            datasets.push({
                label: `Weekly Reports (Avg: ${weeklyAvg.toFixed(1)})`,
                data: weeklyData,
                borderColor: secondaryColor,
                backgroundColor: secondaryColor + '30',
                fill: false,
                tension: 0.4,
                spanGaps: false,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: weeklyData.map(v => v !== null ? secondaryColor : 'transparent'),
                borderWidth: 2,
                borderDash: [5, 3]
            });
        }

        if (this.currentMode === 'monthly') {
            const validMonthly = monthlyData.filter(v => v !== null);
            const monthlyAvg = validMonthly.length > 0
                ? validMonthly.reduce((a, b) => a + b, 0) / validMonthly.length
                : 0;

            datasets.push({
                label: `Monthly Reports (Avg: ${monthlyAvg.toFixed(1)})`,
                data: monthlyData,
                borderColor: accentColor,
                backgroundColor: accentColor + '30',
                fill: true,
                tension: 0.4,
                spanGaps: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: monthlyData.map(v => v !== null ? accentColor : 'transparent'),
                borderWidth: 2
            });
        }

        // Detect light theme
        const isLightTheme = document.body.classList.contains('light-theme');
        const textColor = isLightTheme ? '#1a1a1a' : '#ffffff';
        const gridColor = isLightTheme ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)';
        const gridColorLight = isLightTheme ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)';

        this.chart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        labels: {
                            color: textColor,
                            font: { size: 13, weight: 'bold' },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: isLightTheme ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.9)',
                        titleColor: isLightTheme ? '#1a1a1a' : '#ffffff',
                        bodyColor: isLightTheme ? '#1a1a1a' : '#ffffff',
                        titleFont: { size: 14 },
                        bodyFont: { size: 13 },
                        padding: 12,
                        borderColor: isLightTheme ? 'rgba(0,0,0,0.1)' : 'transparent',
                        borderWidth: isLightTheme ? 1 : 0,
                        callbacks: {
                            label: (context) => {
                                if (context.raw === null) return 'No data';
                                return `${context.dataset.label.split('(')[0]}: ${context.raw}/10`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 10,
                        ticks: {
                            color: isLightTheme ? '#666666' : '#aaaaaa',
                            stepSize: 1,
                            font: { size: 12 }
                        },
                        grid: {
                            color: gridColor
                        }
                    },
                    x: {
                        ticks: {
                            color: isLightTheme ? '#666666' : '#aaaaaa',
                            maxRotation: 45,
                            font: { size: 11 }
                        },
                        grid: {
                            color: gridColorLight
                        }
                    }
                }
            }
        });
    },

    /**
     * Calculate Momentum Score
     * Metric representing user progress with symmetric dampening and logarithmic streak scaling.
     */
    calculateMomentum() {
        const journals = Storage.getJournals();
        if (journals.length === 0) return { score: 0, trend: 'stable' };

        // 1. Sort journals by date (descending)
        const sorted = [...journals].sort((a, b) => new Date(b.date) - new Date(a.date));

        // 2. Calculate Global Average
        const totalRating = sorted.reduce((sum, j) => sum + parseFloat(j.dailyRating), 0);
        const globalAvg = totalRating / sorted.length;

        // 3. Short Term Average (Last 3 entries weighted)
        // Note: Using entries rather than strict dates to ensure "active" momentum is measured
        // even if a user skips a day, though strict dates might be more "honest".
        // User asked for "consecutive days have bigger effect".
        // Let's use strict 3 newest entries, but applying weights is correct.
        const recent = sorted.slice(0, 3);
        let weightedSum = 0;
        let weightTotal = 0;
        const weights = [0.5, 0.3, 0.2]; // Newest to oldest

        recent.forEach((j, i) => {
            weightedSum += parseFloat(j.dailyRating) * weights[i];
            weightTotal += weights[i];
        });
        const shortTermAvg = weightedSum / weightTotal;

        // 4. Moving Average (Last 14 entries)
        const movingSet = sorted.slice(0, 14);
        const movingAvg = movingSet.reduce((sum, j) => sum + parseFloat(j.dailyRating), 0) / movingSet.length;

        // 5. Raw Delta
        let delta = shortTermAvg - movingAvg;

        // 6. Dampening (Symmetric) based on Global Average context
        // If improving (delta > 0) but still below global average -> damp by 0.5
        if (delta > 0 && shortTermAvg < globalAvg) {
            delta *= 0.5;
        }
        // If declining (delta < 0) but still above global average -> damp by 0.5
        if (delta < 0 && shortTermAvg > globalAvg) {
            delta *= 0.5;
        }

        // 7. Calculate Streak (needed for multiplier)
        let streak = 0;
        if (sorted.length > 0) {
            const todayStr = new Date().toISOString().split('T')[0];
            const yesterdayStr = new Date(Date.now() - 86400000).toISOString().split('T')[0];
            const latestDate = sorted[0].date;

            // Streak is active if latest entry is today or yesterday
            if (latestDate === todayStr || latestDate === yesterdayStr) {
                streak = 1;
                let currentDate = new Date(latestDate);

                for (let i = 0; i < sorted.length - 1; i++) {
                    const prevDate = new Date(sorted[i + 1].date);
                    const dayDiff = Math.round((currentDate - prevDate) / (1000 * 60 * 60 * 24));

                    if (dayDiff === 1) {
                        streak++;
                        currentDate = prevDate;
                    } else {
                        break;
                    }
                }
            }
        }

        // 8. Apply Streak Multiplier (Logarithmic)
        // Multiplier grows slowly: Streak 10 -> ~1.5x, Streak 100 -> ~2x
        const streakMultiplier = 1 + (Math.log10(streak + 1) * 0.5);

        // Base scalar 10 makes the number readable (e.g., 0.5 -> 5.0)
        let finalScore = delta * 10 * streakMultiplier;

        // 9. Clamping (-100 to 100) to prevent absurd numbers
        finalScore = Math.max(-100, Math.min(100, finalScore));

        return {
            score: finalScore.toFixed(1),
            trend: finalScore >= 0 ? 'up' : 'down',
            streak: streak // Return streak too, useful for UI
        };
    }
};

window.Statistics = Statistics;
