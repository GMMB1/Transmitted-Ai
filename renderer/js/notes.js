/**
 * Generic Notes Module
 * Uses the /api/notes endpoints on the Flask Backend natively storing into Data/Notes_Data
 */

const Notes = {
    modal: null,
    listModal: null,
    viewModal: null,
    grid: null,
    currentNoteId: null,
    _openedFromList: false,

    init() {
        this.modal = document.getElementById('general-note-modal');
        this.listModal = document.getElementById('general-note-list-modal');
        this.viewModal = document.getElementById('general-note-view-modal');
        this.grid = document.getElementById('general-notes-grid');

        this.bindEvents();
    },

    bindEvents() {
        // Main UI "📝 Note" button → open the notes list directly
        document.getElementById('general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showList();
        });

        // Close Editor
        document.getElementById('close-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this._backFromEditor();
        });

        // Save Note → save then return to list
        document.getElementById('save-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.saveNote();
        });

        // "➕ New Note" inside the list → open blank editor
        document.getElementById('create-new-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hideList();
            this._openedFromList = true;
            this.openEditor();
        });

        // Close History List
        document.getElementById('close-general-note-list')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hideList();
        });

        // Close View modal
        document.getElementById('close-general-note-view')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hideView();
            this.showList();
        });
    },

    openEditor(note = null) {
        if (!this.modal) return;

        const titleInput = document.getElementById('general-note-title');
        const dateInput = document.getElementById('general-note-date');
        const contentInput = document.getElementById('general-note-content');

        if (note) {
            this.currentNoteId = note.id;
            titleInput.value = note.title || '';
            dateInput.value = note.date || '';
            contentInput.value = note.content || '';
        } else {
            this.currentNoteId = null;
            titleInput.value = '';
            dateInput.value = new Date().toISOString().split('T')[0];
            contentInput.value = '';
        }

        this.modal.classList.add('open');
        const overlay = document.getElementById('popup-overlay');
        if (overlay) overlay.classList.add('active');

        setTimeout(() => contentInput.focus(), 100);
    },

    _backFromEditor() {
        if (this.modal) this.modal.classList.remove('open');
        if (this._openedFromList) {
            this._openedFromList = false;
            this.showList();
        } else {
            const overlay = document.getElementById('popup-overlay');
            if (overlay) overlay.classList.remove('active');
        }
    },

    closeEditor() {
        if (this.modal) this.modal.classList.remove('open');
        const overlay = document.getElementById('popup-overlay');
        if (overlay && (!this.listModal || !this.listModal.classList.contains('active'))) {
            overlay.classList.remove('active');
        }
    },

    viewNote(note) {
        if (!this.viewModal) return;
        document.getElementById('view-note-title').textContent = note.title || 'Untitled Note';
        document.getElementById('view-note-date').textContent = note.date || '';
        document.getElementById('view-note-content').textContent = note.content || '';
        this.viewModal.classList.add('open');
        const overlay = document.getElementById('popup-overlay');
        if (overlay) overlay.classList.add('active');
    },

    hideView() {
        if (this.viewModal) this.viewModal.classList.remove('open');
    },

    async saveNote() {
        const title = document.getElementById('general-note-title').value.trim();
        const date = document.getElementById('general-note-date').value.trim();
        const content = document.getElementById('general-note-content').value.trim();

        if (!title && !content) {
            if (typeof Toast !== 'undefined') Toast.warning('Note cannot be completely empty.');
            else alert('Note cannot be completely empty.');
            return;
        }

        const payload = { id: this.currentNoteId, title, date, content };

        try {
            const url = this.currentNoteId ? `/api/notes/${this.currentNoteId}` : '/api/notes';
            const method = this.currentNoteId ? 'PUT' : 'POST';

            const res = await fetch(url, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                if (typeof Toast !== 'undefined') Toast.success('Note saved!');
                // Always return to list after save
                this.modal.classList.remove('open');
                this._openedFromList = false;
                this.showList();
            } else {
                if (typeof Toast !== 'undefined') Toast.error('Failed to save note.');
            }
        } catch (e) {
            console.error('Error saving note:', e);
            if (typeof Toast !== 'undefined') Toast.error('Error connecting to backend.');
        }
    },

    async fetchNotes() {
        try {
            const res = await fetch('/api/notes');
            if (!res.ok) return [];
            return await res.json();
        } catch (e) {
            console.error('Error fetching notes:', e);
            return [];
        }
    },

    async showList() {
        if (!this.listModal || !this.grid) return;

        const overlay = document.getElementById('popup-overlay');
        if (overlay) overlay.classList.add('active');

        this.listModal.classList.add('active');
        this.listModal.style.display = ''; // let CSS class control display (flex, not block)

        this.grid.innerHTML = '<div style="opacity:0.6; padding: 20px;">Loading notes...</div>';

        const notes = await this.fetchNotes();
        notes.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));

        this.grid.innerHTML = '';
        if (notes.length === 0) {
            this.grid.innerHTML = '<div style="opacity:0.6; padding: 20px;">No notes yet. Click ➕ New Note to get started.</div>';
            return;
        }

        notes.forEach(note => {
            const isStruck = !!note.struck;
            const card = document.createElement('div');
            card.className = 'report-card note-card' + (isStruck ? ' note-struck' : '');

            const excerpt = note.content && note.content.length > 100
                ? note.content.substring(0, 100) + '…'
                : note.content;

            card.innerHTML = `
                <div class="report-header">
                    <h4 class="note-card-title">${note.title || 'Untitled Note'}</h4>
                    <span class="report-date">${note.date || 'Unknown Date'}</span>
                </div>
                <div class="report-preview note-card-preview" style="font-size: 0.85rem; opacity: 0.8; margin-top: 10px;">
                    ${excerpt || '<em>No content</em>'}
                </div>
                <div class="report-actions" style="margin-top: 15px; display: flex; gap: 5px; flex-wrap: wrap;">
                    <button class="secondary-btn btn-view" style="padding: 5px 10px; font-size:0.8rem;">👁 View</button>
                    <button class="secondary-btn btn-edit" style="padding: 5px 10px; font-size:0.8rem;">✏ Edit</button>
                    <button class="secondary-btn btn-strike" title="${isStruck ? 'Unstrike' : 'Strike — mark as done'}"
                        style="padding: 5px 10px; font-size:0.8rem; ${isStruck ? 'background:rgba(255,200,0,0.15); color:#f0c040;' : ''}">
                        ${isStruck ? '↩ Unstrike' : '~~' + ' Strike'}
                    </button>
                    <button class="secondary-btn btn-delete" style="padding: 5px 10px; font-size:0.8rem; background:rgba(255,0,0,0.1); color:#ff6b6b;">🗑 Delete</button>
                </div>
            `;

            // View Note (read-only)
            card.querySelector('.btn-view').addEventListener('click', (e) => {
                e.preventDefault();
                this.hideList();
                this.viewNote(note);
            });

            // Edit Note
            card.querySelector('.btn-edit').addEventListener('click', (e) => {
                e.preventDefault();
                this.hideList();
                this._openedFromList = true;
                this.openEditor(note);
            });

            // Strike / Unstrike
            card.querySelector('.btn-strike').addEventListener('click', async (e) => {
                e.preventDefault();
                await this.toggleStruck(note.id);
            });

            // Delete Note
            card.querySelector('.btn-delete').addEventListener('click', async (e) => {
                e.preventDefault();
                if (confirm('Permanently delete this note?')) {
                    try {
                        const res = await fetch(`/api/notes/${note.id}`, { method: 'DELETE' });
                        if (res.ok) {
                            if (typeof Toast !== 'undefined') Toast.success('Deleted note.');
                            this.showList();
                        }
                    } catch (err) {
                        console.error('Failed to delete', err);
                    }
                }
            });

            this.grid.appendChild(card);
        });
    },

    async toggleStruck(id) {
        try {
            const res = await fetch(`/api/notes/${id}/struck`, { method: 'PATCH' });
            if (res.ok) {
                const data = await res.json();
                const msg = data.struck ? 'Struck through — marked as done.' : 'Strike removed.';
                if (typeof Toast !== 'undefined') Toast.success(msg);
                await this.showList(); // await so re-render completes before anything else
            }
        } catch (err) {
            console.error('Strike toggle failed', err);
        }
    },

    hideList() {
        if (this.listModal) {
            this.listModal.classList.remove('active');
            this.listModal.style.display = ''; // clear inline, let CSS handle it
        }
        const overlay = document.getElementById('popup-overlay');
        if (overlay && (!this.modal || !this.modal.classList.contains('open'))) {
            overlay.classList.remove('active');
        }
    }
};

window.Notes = Notes;
