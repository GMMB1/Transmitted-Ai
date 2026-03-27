/**
 * Generic Notes Module
 * Uses the /api/notes endpoints on the Flask Backend natively storing into Data/Notes_Data
 */

const Notes = {
    modal: null,
    listModal: null,
    grid: null,
    currentNoteId: null,

    init() {
        this.modal = document.getElementById('general-note-modal');
        this.listModal = document.getElementById('general-note-list-modal');
        this.grid = document.getElementById('general-notes-grid');

        this.bindEvents();
    },

    bindEvents() {
        // Open Editor from Main UI button
        document.getElementById('general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.openEditor();
        });

        // Close Editor
        document.getElementById('close-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.closeEditor();
        });

        // Save Note
        document.getElementById('save-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.saveNote();
        });

        // Open History List
        document.getElementById('list-general-notes-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.closeEditor();
            this.showList();
        });

        // Add New from History List
        document.getElementById('create-new-general-note-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hideList();
            this.openEditor();
        });

        // Close History List
        document.getElementById('close-general-note-list')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.hideList();
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

    closeEditor() {
        if(this.modal) this.modal.classList.remove('open');
        const overlay = document.getElementById('popup-overlay');
        if (overlay && (!this.listModal || !this.listModal.classList.contains('active'))) {
            overlay.classList.remove('active');
        }
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

        const payload = {
            id: this.currentNoteId,
            title,
            date,
            content
        };

        try {
            const url = this.currentNoteId ? `/api/notes/${this.currentNoteId}` : '/api/notes';
            const method = this.currentNoteId ? 'PUT' : 'POST';

            const res = await fetch(url, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                if (typeof Toast !== 'undefined') Toast.success('Note saved securely!');
                this.closeEditor();
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
        this.listModal.style.display = 'block'; // Fallback if css 'active' doesn't auto-show

        this.grid.innerHTML = '<div style="opacity:0.6; padding: 20px;">Loading notes...</div>';

        const notes = await this.fetchNotes();
        // Sort descending
        notes.sort((a,b) => new Date(b.date || 0) - new Date(a.date || 0));

        this.grid.innerHTML = '';
        if (notes.length === 0) {
            this.grid.innerHTML = '<div style="opacity:0.6; padding: 20px;">No notes found in Data structure.</div>';
            return;
        }

        notes.forEach(note => {
            const card = document.createElement('div');
            card.className = 'report-card';
            
            const excerpt = note.content && note.content.length > 80 ? note.content.substring(0,80) + '...' : note.content;

            card.innerHTML = `
                <div class="report-header">
                    <h4>${note.title || 'Untitled Note'}</h4>
                    <span class="report-date">${note.date || 'Unknown Date'}</span>
                </div>
                <div class="report-preview" style="font-size: 0.85rem; opacity: 0.8; margin-top: 10px;">
                    ${excerpt || '<em>No details mapped</em>'}
                </div>
                <div class="report-actions" style="margin-top: 15px; display: flex; gap: 5px;">
                    <button class="secondary-btn btn-edit" style="padding: 5px 10px; font-size:0.8rem;">Edit</button>
                    <button class="secondary-btn btn-delete" style="padding: 5px 10px; font-size:0.8rem; background:rgba(255,0,0,0.1); color:#ff6b6b;">Delete</button>
                </div>
            `;

            // Edit Note
            card.querySelector('.btn-edit').addEventListener('click', (e) => {
                e.preventDefault();
                this.hideList();
                this.openEditor(note);
            });

            // Delete Note
            card.querySelector('.btn-delete').addEventListener('click', async (e) => {
                e.preventDefault();
                if (confirm('Permanently delete this note?')) {
                    try {
                        const res = await fetch(`/api/notes/${note.id}`, { method: 'DELETE' });
                        if (res.ok) {
                            if (typeof Toast !== 'undefined') Toast.success('Deleted note.');
                            this.showList(); // refresh
                        }
                    } catch(err) {
                        console.error('Failed to delete', err);
                    }
                }
            });

            this.grid.appendChild(card);
        });
    },

    hideList() {
        if(this.listModal) {
            this.listModal.classList.remove('active');
            this.listModal.style.display = 'none';
        }
        const overlay = document.getElementById('popup-overlay');
        if (overlay && (!this.modal || !this.modal.classList.contains('open'))) {
            overlay.classList.remove('active');
        }
    }
};

window.Notes = Notes;
