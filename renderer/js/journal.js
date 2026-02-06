/**
 * Journal Module - Handles journal CRUD operations
 */

const Journal = {
    // Emoji mapping based on rating
    emojiMap: [
        { emoji: '😩', min: 0, max: 1.5 },
        { emoji: '😔', min: 1.5, max: 3 },
        { emoji: '😕', min: 3, max: 4.5 },
        { emoji: '😐', min: 4.5, max: 6 },
        { emoji: '😊', min: 6, max: 7.5 },
        { emoji: '😃', min: 7.5, max: 9 },
        { emoji: '🤩', min: 9, max: 10.1 }
    ],

    /**
     * Initialize journal module
     */
    init() {
        // Form elements
        this.titleInput = document.getElementById('journal-title');
        this.dateInput = document.getElementById('journal-date');
        this.detailsInput = document.getElementById('journal-details');
        this.ratingSlider = document.getElementById('daily-rating');
        this.ratingValue = document.getElementById('rating-value');
        this.countDisplay = document.getElementById('journal-count');

        // Emoji elements
        this.leftEmoji = document.getElementById('left-emoji');
        this.centerEmoji = document.getElementById('center-emoji');
        this.rightEmoji = document.getElementById('right-emoji');

        // Popup elements
        this.popup = document.getElementById('journals-popup');
        this.listContainer = document.getElementById('journals-list');

        // Set default date to today
        this.dateInput.value = new Date().toISOString().split('T')[0];

        // Setup event listeners
        this.setupListeners();
        this.updateCount();
        this.updateEmojis(5);

        // Auto-apply default template on load
        setTimeout(() => this.applyDefaultTemplate(), 100);
    },

    /**
     * Setup event listeners
     */
    setupListeners() {
        // Rating slider
        this.ratingSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.ratingValue.textContent = `${value.toFixed(2)}/10`;
            this.updateEmojis(value);
        });

        // Add journal button
        document.getElementById('add-journal-btn').addEventListener('click', () => this.add());

        // Show journals button
        document.getElementById('show-journals-btn').addEventListener('click', () => this.showPopup());

        // Close popup
        document.getElementById('close-journals-popup').addEventListener('click', () => this.hidePopup());

        // Sort controls
        document.getElementById('sort-type').addEventListener('change', () => this.renderList());
        document.getElementById('sort-direction').addEventListener('change', () => this.renderList());

        // Export controls
        document.getElementById('export-mode').addEventListener('change', (e) => {
            const dateInputs = document.getElementById('export-date-inputs');
            dateInputs.classList.toggle('hidden', e.target.value !== 'range');
        });
        document.getElementById('export-btn').addEventListener('click', () => this.exportCSV());
    },

    /**
     * Get emoji for rating value
     */
    getEmoji(rating) {
        for (const item of this.emojiMap) {
            if (rating >= item.min && rating < item.max) {
                return item.emoji;
            }
        }
        return '😊';
    },

    /**
     * Get emoji index for rating
     */
    getEmojiIndex(rating) {
        for (let i = 0; i < this.emojiMap.length; i++) {
            if (rating >= this.emojiMap[i].min && rating < this.emojiMap[i].max) {
                return i;
            }
        }
        return 4;
    },

    /**
     * Update emoji display based on rating
     */
    updateEmojis(rating) {
        const index = this.getEmojiIndex(rating);

        this.leftEmoji.textContent = index > 0 ? this.emojiMap[index - 1].emoji : '';
        this.centerEmoji.textContent = this.emojiMap[index].emoji;
        this.rightEmoji.textContent = index < this.emojiMap.length - 1 ? this.emojiMap[index + 1].emoji : '';
    },

    /**
     * Update journal count display
     */
    updateCount() {
        this.countDisplay.textContent = Storage.getJournals().length;
    },

    /**
     * Add a new journal entry
     */
    async add() {
        const title = this.titleInput.value.trim();
        const date = this.dateInput.value;
        const details = this.detailsInput.value.trim();
        const rating = parseFloat(this.ratingSlider.value);

        if (!date || !details) {
            Toast.warning('Please fill in the date and journal details!');
            return;
        }

        /*
        if (Storage.journalExistsForDate(date)) {
            Toast.error('A journal entry for this date already exists!');
            return;
        }
        */

        const journal = {
            title: title,
            date,
            details,
            dailyRating: rating,
            emoji: this.getEmoji(rating),
            createdAt: new Date().toISOString()
        };

        try {
            await Storage.addJournal(journal);
            this.clearForm();
            this.updateCount();
            Toast.success('Journal saved successfully!');
        } catch (error) {
            console.error(error);
            Toast.error('Failed to save journal.');
        }
    },

    /**
     * Clear form inputs
     */
    clearForm() {
        this.titleInput.value = '';
        this.detailsInput.value = '';
        this.ratingSlider.value = 5;
        this.ratingValue.textContent = '5.00/10';
        this.updateEmojis(5);
        // Keep date as today
        this.dateInput.value = new Date().toISOString().split('T')[0];

        this.applyDefaultTemplate();
    },

    /**
     * Apply default template if exists
     */
    applyDefaultTemplate() {
        const templates = Storage.getTemplates();
        if (templates && templates.length > 0) {
            // Handle both migrated and old formats just in case, though migration should run first
            const defaultTemplate = templates.find(t => typeof t === 'object' && t.isDefault);
            if (defaultTemplate) {
                this.detailsInput.value = defaultTemplate.content;
            }
        }
    },

    /**
     * Show journals popup
     */
    showPopup() {
        document.getElementById('popup-overlay').classList.add('active');
        this.popup.classList.add('open');
        this.renderList();
    },

    /**
     * Hide journals popup
     */
    hidePopup() {
        document.getElementById('popup-overlay').classList.remove('active');
        this.popup.classList.remove('open');
    },

    /**
     * Render sorted journal list
     */
    renderList() {
        this.renderJournals();
    },

    /**
     * Render journals with optional search/favorites filter
     */
    renderJournals(searchQuery = null, favoritesOnly = false) {
        const sortType = document.getElementById('sort-type').value;
        const direction = document.getElementById('sort-direction').value;

        let journals = [...Storage.getJournals()];

        // Filter by favorites
        if (favoritesOnly) {
            const favorites = Storage.getFavorites();
            journals = journals.filter(j => favorites.includes(j.date));
        }

        // Filter by search query
        if (searchQuery) {
            const q = searchQuery.toLowerCase();
            journals = journals.filter(j => {
                const title = (j.title || '').toLowerCase();
                const details = (j.details || '').toLowerCase();
                const date = (j.date || '');
                return title.includes(q) || details.includes(q) || date.includes(q);
            });
        }

        // Sort
        journals.sort((a, b) => {
            let compare = 0;
            switch (sortType) {
                case 'date':
                    compare = new Date(a.date) - new Date(b.date);
                    break;
                case 'rating':
                    compare = a.dailyRating - b.dailyRating;
                    break;
                case 'title':
                    compare = a.title.localeCompare(b.title);
                    break;
            }
            return direction === 'desc' ? -compare : compare;
        });

        this.listContainer.innerHTML = '';

        if (journals.length === 0) {
            const msg = favoritesOnly ? 'No favorite journals yet. Star some journals!'
                : searchQuery ? 'No journals match your search.'
                    : 'No journals yet. Start writing!';
            this.listContainer.innerHTML = `<p class="placeholder-msg">${msg}</p>`;
            return;
        }

        journals.forEach((journal, index) => {
            const originalIndex = Storage.getJournals().findIndex(j => j.date === journal.date && j.id === journal.id);
            const card = this.createCard(journal, originalIndex);
            
            // Highlight first result when searching
            if (searchQuery && index === 0) {
                card.classList.add('search-highlight');
                // Scroll to first result after render
                setTimeout(() => {
                    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 100);
            }
            
            this.listContainer.appendChild(card);
        });
    },

    /**
     * Toggle favorite status
     */
    toggleFavorite(date) {
        Storage.toggleFavorite(date);
    },

    /**
     * Check if journal is favorite
     */
    isFavorite(date) {
        return Storage.isFavorite(date);
    },

    /**
     * Create a journal card element
     */
    createCard(journal, index) {
        const card = document.createElement('div');
        card.className = 'journal-card';

        const preview = journal.details.split('\n')[0].substring(0, 60) + '...';
        const isFav = this.isFavorite(journal.date);

        card.innerHTML = `
            <button class="favorite-btn ${isFav ? 'active' : ''}" title="Toggle Favorite">
                ${isFav ? '⭐' : '☆'}
            </button>
            <h3>${journal.title}</h3>
            <div class="date">${journal.date}</div>
            <div class="emoji-display">${journal.emoji}</div>
            <div class="rating">${journal.dailyRating}/10</div>
            <div class="preview">${preview}</div>
            <div class="card-actions">
                <button class="secondary-btn view-btn">View</button>
                <button class="primary-btn edit-btn">Edit</button>
                <button class="danger-btn delete-btn">Delete</button>
            </div>
        `;

        // Event listeners
        card.querySelector('.favorite-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleFavorite(journal.date);
            this.renderJournals();
        });
        card.querySelector('.view-btn').addEventListener('click', () => this.view(index));
        card.querySelector('.edit-btn').addEventListener('click', () => this.edit(index));
        card.querySelector('.delete-btn').addEventListener('click', () => this.confirmDelete(index));

        return card;
    },

    /**
     * View journal details
     */
    view(index) {
        const journal = Storage.getJournals()[index];
        if (!journal) return;
        this.openViewModal(journal);
    },

    /**
     * View journal by date (for Random Day feature)
     */
    viewByDate(date) {
        const journals = Storage.getJournals();
        const journal = journals.find(j => j.date === date);
        if (journal) {
            this.openViewModal(journal);
        } else {
            Toast.error('Journal entry not found.');
        }
    },

    /**
     * Open View Modal Helper
     */
    openViewModal(journal) {
        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 800px; max-height: 80vh; overflow-y: auto; text-align: center;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <span style="font-size: 1.5rem;">${journal.date}</span>
                    <h2 style="color: var(--secondary-color); font-size: 2rem; flex: 1; margin: 0 20px;">${journal.title}</h2>
                    <span style="font-size: 2rem;">${journal.emoji} ${journal.dailyRating}/10</span>
                </div>
                <div style="white-space: pre-wrap; text-align: center; direction: rtl; font-size: 1.2rem; line-height: 1.8;">
                    ${journal.details}
                </div>
                <button class="secondary-btn" style="margin-top: 25px;">Close</button>
            </div>
        `;

        overlay.querySelector('button').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });

        document.body.appendChild(overlay);
    },

    /**
     * Edit journal
     */
    edit(index) {
        const journal = Storage.getJournals()[index];
        if (!journal) return;

        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 700px; text-align: left;">
                <h2 style="text-align: center; margin-bottom: 20px;">Edit Journal</h2>
                <input type="text" id="edit-title" value="${journal.title}" placeholder="Title" style="margin-bottom: 10px;">
                <input type="date" id="edit-date" value="${journal.date}" style="margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 15px; margin: 15px 0;">
                    <span id="edit-emoji" style="font-size: 3rem;">${journal.emoji}</span>
                    <input type="range" id="edit-rating" min="0" max="10" step="0.25" value="${journal.dailyRating}" style="flex: 1;">
                    <span id="edit-rating-display" style="font-size: 1.5rem; min-width: 80px;">${journal.dailyRating}/10</span>
                </div>
                <textarea id="edit-details" style="height: 150px; margin-bottom: 15px;">${journal.details}</textarea>
                <div class="modal-actions">
                    <button id="save-edit-btn" class="primary-btn">Save Changes</button>
                    <button id="cancel-edit-btn" class="secondary-btn">Cancel</button>
                </div>
            </div>
        `;

        const ratingSlider = overlay.querySelector('#edit-rating');
        const ratingDisplay = overlay.querySelector('#edit-rating-display');
        const emojiDisplay = overlay.querySelector('#edit-emoji');

        ratingSlider.addEventListener('input', () => {
            const val = parseFloat(ratingSlider.value);
            ratingDisplay.textContent = `${val}/10`;
            emojiDisplay.textContent = this.getEmoji(val);
        });

        overlay.querySelector('#save-edit-btn').addEventListener('click', async () => {
            const updatedJournal = {
                ...journal,
                title: overlay.querySelector('#edit-title').value,
                date: overlay.querySelector('#edit-date').value,
                dailyRating: parseFloat(ratingSlider.value),
                details: overlay.querySelector('#edit-details').value,
                emoji: emojiDisplay.textContent
            };

            await Storage.updateJournal(index, updatedJournal);
            this.renderList();
            overlay.remove();
        });

        overlay.querySelector('#cancel-edit-btn').addEventListener('click', () => overlay.remove());

        document.body.appendChild(overlay);
    },

    /**
     * Confirm delete
     */
    confirmDelete(index) {
        const modal = document.getElementById('confirm-modal');
        const message = document.getElementById('confirm-message');

        message.textContent = 'Are you sure you want to delete this journal?';
        modal.classList.remove('hidden');

        const yesBtn = document.getElementById('confirm-yes');
        const noBtn = document.getElementById('confirm-no');

        // Replace buttons to clear previous listeners
        const newYesBtn = yesBtn.cloneNode(true);
        const newNoBtn = noBtn.cloneNode(true);
        yesBtn.replaceWith(newYesBtn);
        noBtn.replaceWith(newNoBtn);

        const cleanup = () => {
            modal.classList.add('hidden');
        };

        newYesBtn.addEventListener('click', async () => {
            await Storage.deleteJournal(index);
            this.renderList();
            this.updateCount();
            cleanup();
        });

        newNoBtn.addEventListener('click', cleanup);
    },

    /**
     * Export journals to CSV
     */
    exportCSV() {
        const mode = document.getElementById('export-mode').value;
        let data = Storage.getJournals();

        if (mode === 'range') {
            const start = document.getElementById('export-start').value;
            const end = document.getElementById('export-end').value;

            if (!start || !end) {
                Toast.warning('Please select start and end dates.');
                return;
            }

            data = Storage.getJournalsByDateRange(new Date(start), new Date(end));

            if (data.length === 0) {
                Toast.info('No journals found in this date range.');
                return;
            }
        }

        const csv = 'Title,Date,Rating,Details\n' +
            data.map(j =>
                `"${j.title}",${j.date},${j.dailyRating},"${j.details.replace(/"/g, '""').replace(/\n/g, ' ')}"`
            ).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = mode === 'range'
            ? `journals_${document.getElementById('export-start').value}_to_${document.getElementById('export-end').value}.csv`
            : 'all_journals.csv';
        link.click();
        URL.revokeObjectURL(url);
    }
};

window.Journal = Journal;
