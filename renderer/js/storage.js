/**
 * Storage Module - Handles data persistence via Electron IPC
 */

const Storage = {
    // Data stores
    journals: [],
    weeklyReports: [],
    monthlyReports: [],
    templates: [],
    quickNotes: '',
    favorites: [],
    settings: {
        theme: 'default',
        lastSuggestedDates: [] // Track last 10 suggested dates
    },

    /**
     * Initialize storage - load data from disk or localStorage
     */
    async init() {
        try {
            // Try to load journals from Flask API first
            const apiJournals = await this.fetchJournalsFromAPI();
            if (apiJournals && apiJournals.length >= 0) {
                // Map API format (mood) to UI format (dailyRating)
                this.journals = apiJournals.map(j => ({
                    ...j,
                    dailyRating: j.mood || 5
                }));
                console.log('Journals loaded from Flask API (psychoanalytical.json)');
            }
            
            // Load other data from localStorage
            const localData = localStorage.getItem('daily_productivity_full_data');
            if (localData) {
                const data = JSON.parse(localData);
                // Don't overwrite journals from API
                this.weeklyReports = data.weeklyReports || [];
                this.monthlyReports = data.monthlyReports || [];
                this.templates = data.templates || [];
                this.quickNotes = data.quickNotes || '';
                this.favorites = data.favorites || [];
                this.settings = data.settings || { theme: 'default' };
            }
        } catch (error) {
            console.error('Error loading data:', error);
            // Fallback to localStorage
            await this.migrateFromLocalStorage();
        }
    },

    /**
     * Fetch journals from Flask API
     */
    async fetchJournalsFromAPI() {
        try {
            const response = await fetch('/api/psycho/entries');
            if (response.ok) {
                return await response.json();
            }
        } catch (e) {
            console.warn('Could not fetch from API, falling back to localStorage');
        }
        return null;
    },

    /**
     * Load data from a data object
     */
    loadFromObject(data) {
        this.journals = data.journals || [];
        this.weeklyReports = data.weeklyReports || [];
        this.monthlyReports = data.monthlyReports || [];

        // Template Migration (String -> Object, Add name field)
        this.templates = (data.templates || []).map((t, index) => {
            if (typeof t === 'string') {
                return {
                    id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
                    name: `Template ${index + 1}`,
                    content: t,
                    isDefault: false
                };
            }
            // Add name if missing
            if (!t.name) {
                t.name = `Template ${index + 1}`;
            }
            return t;
        });

        this.quickNotes = data.quickNotes || '';
        this.favorites = data.favorites || [];
        this.settings = data.settings || { theme: 'default' };
    },

    /**
     * Migrate data from old localStorage format
     */
    async migrateFromLocalStorage() {
        try {
            const oldJournals = localStorage.getItem('Journals');
            const oldWeeklyReports = localStorage.getItem('WeeklyReports');
            const oldTemplates = localStorage.getItem('templates');
            const oldTheme = localStorage.getItem('selectedTheme');

            let migrated = false;

            if (oldJournals) {
                this.journals = JSON.parse(oldJournals);
                migrated = true;
            }

            if (oldWeeklyReports) {
                this.weeklyReports = JSON.parse(oldWeeklyReports);
                migrated = true;
            }

            if (oldTemplates) {
                this.templates = JSON.parse(oldTemplates);
                migrated = true;
            }

            if (oldTheme) {
                this.settings.theme = oldTheme;
                migrated = true;
            }

            if (migrated) {
                await this.save();
                console.log('Migration complete!');
            }
        } catch (error) {
            console.error('Error migrating from localStorage:', error);
        }
    },

    /**
     * Save all data to disk or localStorage
     */
    async save() {
        try {
            const data = {
                journals: this.journals,
                weeklyReports: this.weeklyReports,
                monthlyReports: this.monthlyReports,
                templates: this.templates,
                quickNotes: this.quickNotes,
                favorites: this.favorites,
                settings: this.settings
            };

            if (window.electronAPI) {
                // Desktop Mode
                const result = await window.electronAPI.saveData(data);
                if (!result.success) {
                    console.error('Save failed:', result.error);
                }
            } else {
                // Web/Mobile Mode
                localStorage.setItem('daily_productivity_full_data', JSON.stringify(data));
                console.log('Saved to localStorage (Web Mode)');
            }
        } catch (error) {
            console.error('Error saving data:', error);
        }
    },

    // ==================== Journals ====================
    async addJournal(journal) {
        // POST to Flask API
        try {
            const response = await fetch('/api/psycho/entries', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(journal)
            });
            if (response.ok) {
                const saved = await response.json();
                // Add to local cache with UI format
                this.journals.push({ ...saved, dailyRating: saved.mood || journal.dailyRating });
                await this.saveLocalData();
                console.log('Journal saved to psychoanalytical.json');
                return true;
            }
        } catch (e) {
            console.error('Failed to save journal to API:', e);
        }
        // Fallback: save locally
        this.journals.push(journal);
        await this.saveLocalData();
        return false;
    },

    async updateJournal(index, journal) {
        if (index >= 0 && index < this.journals.length) {
            const id = this.journals[index].id;
            try {
                const response = await fetch(`/api/psycho/entries/${id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(journal)
                });
                if (response.ok) {
                    const updated = await response.json();
                    this.journals[index] = { ...updated, dailyRating: updated.mood || journal.dailyRating };
                    await this.saveLocalData();
                    return;
                }
            } catch (e) {
                console.error('Failed to update journal via API:', e);
            }
            this.journals[index] = journal;
            await this.saveLocalData();
        }
    },

    async deleteJournal(index) {
        if (index >= 0 && index < this.journals.length) {
            const id = this.journals[index].id;
            try {
                await fetch(`/api/psycho/entries/${id}`, { method: 'DELETE' });
            } catch (e) {
                console.error('Failed to delete journal via API:', e);
            }
            this.journals.splice(index, 1);
            await this.saveLocalData();
        }
    },

    /**
     * Save non-journal data to localStorage (journals are saved to API)
     */
    async saveLocalData() {
        try {
            const data = {
                journals: this.journals, // Keep a local cache
                weeklyReports: this.weeklyReports,
                monthlyReports: this.monthlyReports,
                templates: this.templates,
                quickNotes: this.quickNotes,
                favorites: this.favorites,
                settings: this.settings
            };
            localStorage.setItem('daily_productivity_full_data', JSON.stringify(data));
        } catch (error) {
            console.error('Error saving local data:', error);
        }
    },


    getJournals() {
        return this.journals;
    },

    getJournalsByDateRange(startDate, endDate) {
        return this.journals.filter(j => {
            const date = new Date(j.date);
            return date >= startDate && date <= endDate;
        });
    },

    journalExistsForDate(date) {
        return this.journals.some(j => j.date === date);
    },

    // ==================== Weekly Reports ====================
    addWeeklyReport(report) {
        const existingIndex = this.weeklyReports.findIndex(
            r => r.year === report.year && r.month === report.month && r.week === report.week
        );
        if (existingIndex !== -1) {
            this.weeklyReports[existingIndex] = report;
        } else {
            this.weeklyReports.push(report);
        }
        this.save();
    },

    deleteWeeklyReport(index) {
        if (index >= 0 && index < this.weeklyReports.length) {
            this.weeklyReports.splice(index, 1);
            this.save();
        }
    },

    getWeeklyReports() {
        return this.weeklyReports;
    },

    getWeeklyReport(year, month, week) {
        return this.weeklyReports.find(
            r => r.year === year && r.month === month && r.week === week
        );
    },

    // ==================== Templates ====================
    // ==================== Templates ====================
    addTemplate(name, content) {
        const template = {
            id: Date.now().toString(),
            name: name || 'Untitled Template',
            content: content,
            isDefault: false
        };
        this.templates.push(template);
        this.save();
    },

    updateTemplate(id, name, content) {
        const index = this.templates.findIndex(t => t.id === id);
        if (index !== -1) {
            this.templates[index].name = name || this.templates[index].name || 'Untitled Template';
            this.templates[index].content = content;
            this.save();
        }
    },

    setTemplateDefault(id) {
        // Unset all others
        this.templates.forEach(t => t.isDefault = false);

        // Set new default (if id provided)
        if (id) {
            const index = this.templates.findIndex(t => t.id === id);
            if (index !== -1) {
                this.templates[index].isDefault = true;
            }
        }
        this.save();
    },

    deleteTemplate(id) {
        const index = this.templates.findIndex(t => t.id === id);
        if (index !== -1) {
            this.templates.splice(index, 1);
            this.save();
        }
    },

    getTemplates() {
        return this.templates;
    },

    // ==================== Settings ====================
    setTheme(theme) {
        this.settings.theme = theme;
        this.save();
    },

    getTheme() {
        return this.settings.theme;
    },

    setNeon(neon) {
        this.settings.neon = neon;
        this.save();
    },

    getNeon() {
        return this.settings.neon || 'none';
    },

    setPattern(pattern) {
        this.settings.pattern = pattern;
        this.save();
    },

    getPattern() {
        return this.settings.pattern || 'none';
    },

    // ==================== Monthly Reports ====================
    addMonthlyReport(report) {
        const existingIndex = this.monthlyReports.findIndex(
            r => r.year === report.year && r.month === report.month
        );
        if (existingIndex !== -1) {
            this.monthlyReports[existingIndex] = report;
        } else {
            this.monthlyReports.push(report);
        }
        this.save();
    },

    deleteMonthlyReport(index) {
        if (index >= 0 && index < this.monthlyReports.length) {
            this.monthlyReports.splice(index, 1);
            this.save();
        }
    },

    getMonthlyReports() {
        return this.monthlyReports;
    },

    getMonthlyReport(year, month) {
        return this.monthlyReports.find(r => r.year === year && r.month === month);
    },

    // ==================== Quick Notes ====================
    setQuickNotes(text) {
        this.quickNotes = text;
        this.save();
    },

    getQuickNotes() {
        return this.quickNotes;
    },

    // ==================== Favorites ====================
    toggleFavorite(date) {
        const index = this.favorites.indexOf(date);
        if (index === -1) {
            this.favorites.push(date);
        } else {
            this.favorites.splice(index, 1);
        }
        this.save();
    },

    isFavorite(date) {
        return this.favorites.includes(date);
    },

    getFavorites() {
        return this.favorites;
    },

    // ==================== Random Day Logic ====================
    /**
     * Get a random journal date, excluding recently suggested ones
     */
    getRandomJournalDate() {
        if (this.journals.length === 0) return null;

        const allDates = this.journals.map(j => j.date);
        let eligibleDates = [];

        // If we have enough journals, exclude the recently suggested ones
        if (allDates.length > 10) {
            const excluded = this.settings.lastSuggestedDates || [];
            eligibleDates = allDates.filter(date => !excluded.includes(date));

            // Fallback if somehow everything is excluded (shouldn't happen with logic, but safety)
            if (eligibleDates.length === 0) eligibleDates = allDates;
        } else {
            eligibleDates = allDates;
        }

        const randomIndex = Math.floor(Math.random() * eligibleDates.length);
        const selectedDate = eligibleDates[randomIndex];

        // Update exclusion list
        this.updateSuggestedHistory(selectedDate);

        return selectedDate;
    },

    updateSuggestedHistory(date) {
        if (!this.settings.lastSuggestedDates) this.settings.lastSuggestedDates = [];

        // Add new date
        this.settings.lastSuggestedDates.push(date);

        // Keep only last 10
        if (this.settings.lastSuggestedDates.length > 10) {
            this.settings.lastSuggestedDates.shift();
        }

        this.save();
    },

    // ==================== Backup & Restore ====================
    exportAllData() {
        return JSON.stringify({
            journals: this.journals,
            weeklyReports: this.weeklyReports,
            monthlyReports: this.monthlyReports,
            templates: this.templates,
            quickNotes: this.quickNotes,
            favorites: this.favorites,
            settings: this.settings,
            exportedAt: new Date().toISOString()
        }, null, 2);
    },

    async importData(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            if (data.journals) this.journals = data.journals;
            if (data.weeklyReports) this.weeklyReports = data.weeklyReports;
            if (data.monthlyReports) this.monthlyReports = data.monthlyReports;
            if (data.templates) this.templates = data.templates;
            if (data.quickNotes !== undefined) this.quickNotes = data.quickNotes;
            if (data.favorites) this.favorites = data.favorites;
            if (data.settings) this.settings = data.settings;

            await this.save();
            return { success: true, message: 'Data imported successfully!' };
        } catch (error) {
            return { success: false, message: 'Invalid backup file format.' };
        }
    }
};

// Make Storage globally available
window.Storage = Storage;

