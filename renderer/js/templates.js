/**
 * Templates Module - Handles template management
 */

const Templates = {
    popup: null,
    input: null,
    nameInput: null,
    list: null,
    editingId: null, // Track which template is being edited
    currentTemplateIndex: -1, // Track current template for cycling

    /**
     * Initialize templates module
     */
    init() {
        this.popup = document.getElementById('template-popup');
        this.input = document.getElementById('template-input');
        this.nameInput = document.getElementById('template-name-input');
        this.list = document.getElementById('template-list');

        // Event listeners
        document.getElementById('manage-templates-btn').addEventListener('click', () => this.show());
        document.getElementById('close-template-popup').addEventListener('click', () => this.hide());
        document.getElementById('save-template-btn').addEventListener('click', () => this.save());
        document.getElementById('apply-template-btn').addEventListener('click', () => this.applyNextTemplate());

        // New Cancel Edit listener (will be added to HTML next)
        const cancelBtn = document.getElementById('cancel-edit-btn');
        if (cancelBtn) cancelBtn.addEventListener('click', () => this.cancelEdit());
    },

    /**
     * Show template manager popup
     */
    show() {
        this.popup.classList.add('open');
        this.cancelEdit(); // Reset state
        this.render();
    },

    /**
     * Hide template manager popup
     */
    hide() {
        this.popup.classList.remove('open');
        this.cancelEdit();
    },

    /**
     * Save or Update a template
     */
    save() {
        const name = this.nameInput.value.trim();
        const text = this.input.value.trim();

        if (!name) {
            Toast.warning('Template name cannot be empty!');
            return;
        }
        if (!text) {
            Toast.warning('Template content cannot be empty!');
            return;
        }

        if (this.editingId) {
            // Update existing
            Storage.updateTemplate(this.editingId, name, text);
            Toast.success('Template updated!');
        } else {
            // Create new
            Storage.addTemplate(name, text);
            Toast.success('Template saved!');
        }

        this.cancelEdit();
        this.render();
    },

    /**
     * Start editing a template
     */
    startEdit(template) {
        this.editingId = template.id;
        this.nameInput.value = template.name || '';
        this.input.value = template.content;

        // Update UI state
        document.getElementById('save-template-btn').textContent = 'Update Template';
        document.getElementById('template-editor-title').textContent = '✏️ Edit Template';
        document.getElementById('cancel-edit-btn').classList.remove('hidden');
        this.nameInput.focus();
    },

    /**
     * Cancel edit mode
     */
    cancelEdit() {
        this.editingId = null;
        this.nameInput.value = '';
        this.input.value = '';

        // Reset UI state
        document.getElementById('save-template-btn').textContent = 'Save Template';
        const title = document.getElementById('template-editor-title');
        if (title) title.textContent = '✨ Create New';
        const cancelBtn = document.getElementById('cancel-edit-btn');
        if (cancelBtn) cancelBtn.classList.add('hidden');
    },

    /**
     * Delete a template by ID
     */
    delete(id) {
        CustomAlert.confirm('Delete this template?', (confirmed) => {
            if (confirmed) {
                if (this.editingId === id) this.cancelEdit();
                Storage.deleteTemplate(id);
                this.render();
            }
        });
    },

    /**
     * Toggle default status
     */
    toggleDefault(id) {
        Storage.setTemplateDefault(id);
        this.render();
        Toast.info('Default template set');
    },

    /**
     * Apply a specific template content
     */
    apply(content) {
        document.getElementById('journal-details').value = content;
        this.hide();
        Toast.success('Template applied');
    },

    /**
     * Apply next template (cycles through all templates)
     * First click: applies default template
     * Subsequent clicks: cycles through other templates
     */
    applyNextTemplate() {
        const templates = Storage.getTemplates();
        if (templates.length === 0) {
            Toast.info('No templates available.');
            return;
        }

        // First click - start with default or first template
        if (this.currentTemplateIndex === -1) {
            const defaultIndex = templates.findIndex(t => t.isDefault);
            if (defaultIndex !== -1) {
                this.currentTemplateIndex = defaultIndex;
            } else {
                this.currentTemplateIndex = 0;
            }
        } else {
            // Subsequent clicks - cycle to next template
            this.currentTemplateIndex = (this.currentTemplateIndex + 1) % templates.length;
        }

        const template = templates[this.currentTemplateIndex];
        document.getElementById('journal-details').value = template.content;

        // Show lightweight tooltip above button
        const templateName = template.name || 'Untitled Template';
        this.showTemplateTooltip(`📝 ${templateName}`);
    },

    /**
     * Show a lightweight tooltip above Apply Template button
     */
    showTemplateTooltip(text) {
        const btn = document.getElementById('apply-template-btn');
        if (!btn) return;

        // Remove existing tooltip if any
        const existing = document.querySelector('.template-tooltip');
        if (existing) existing.remove();

        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'template-tooltip';
        tooltip.textContent = text;

        // Position above button
        btn.style.position = 'relative';
        btn.appendChild(tooltip);

        // Auto-remove after 2 seconds
        setTimeout(() => {
            tooltip.classList.add('fade-out');
            setTimeout(() => tooltip.remove(), 300);
        }, 2000);
    },

    /**
     * Render template list
     */
    render() {
        const templates = Storage.getTemplates();
        this.list.innerHTML = '';

        if (templates.length === 0) {
            this.list.innerHTML = '<p class="placeholder-msg">No templates saved.</p>';
            return;
        }

        templates.forEach(t => {
            const item = document.createElement('div');
            item.className = `template-card ${t.isDefault ? 'default' : ''}`;
            if (this.editingId === t.id) item.classList.add('editing');

            // Template Name (Title)
            const nameEl = document.createElement('div');
            nameEl.className = 'template-name';
            nameEl.textContent = t.name || 'Untitled Template';

            // Template Content Preview (smaller)
            const preview = document.createElement('div');
            preview.className = 'template-preview';
            preview.textContent = t.content.substring(0, 40) + (t.content.length > 40 ? '...' : '');

            // Actions Container
            const actions = document.createElement('div');
            actions.className = 'template-actions';

            // Star (Default) Btn
            const starBtn = document.createElement('button');
            starBtn.className = `icon-btn star-btn ${t.isDefault ? 'active' : ''}`;
            starBtn.innerHTML = t.isDefault ? '★' : '☆';
            starBtn.title = 'Set as Default';
            starBtn.onclick = (e) => { e.stopPropagation(); this.toggleDefault(t.id); };

            // Edit Btn
            const editBtn = document.createElement('button');
            editBtn.className = 'icon-btn edit-btn';
            editBtn.innerHTML = '✏️';
            editBtn.title = 'Edit';
            editBtn.onclick = (e) => { e.stopPropagation(); this.startEdit(t); };

            // Delete Btn
            const delBtn = document.createElement('button');
            delBtn.className = 'icon-btn del-btn';
            delBtn.innerHTML = '🗑️';
            delBtn.title = 'Delete';
            delBtn.onclick = (e) => { e.stopPropagation(); this.delete(t.id); };

            // Apply Btn
            const applyBtn = document.createElement('button');
            applyBtn.className = 'mini-btn primary';
            applyBtn.textContent = 'Use';
            applyBtn.onclick = (e) => { e.stopPropagation(); this.apply(t.content); };

            actions.append(starBtn, editBtn, delBtn, applyBtn);
            item.append(nameEl, preview, actions);

            // Allow clicking card to edit
            item.onclick = () => this.startEdit(t);

            this.list.appendChild(item);
        });
    }
};

window.Templates = Templates;
