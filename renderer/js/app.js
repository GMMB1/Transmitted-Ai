/**
 * Main App Module - Initializes all modules and handles global UI
 */

const App = {
    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing Daily Productivity App...');

        // Load data from disk first
        await Storage.init();

        // Initialize all modules
        Themes.init();
        // Templates.init(); // Removed
        Journal.init();
        Weekly.init();
        Monthly.init();
        Statistics.init();

        // Setup global UI
        this.setupSideMenu();
        this.setupNavigation();
        // this.setupStreak(); // Removed
        this.setupRandomDay();
        this.setupAnalysis(); // New
        this.setupQuickNotes();
        this.setupSearch();
        this.setupBackup();
        this.setupResizableTextareas();
        this.setupPrivacyToggle();

        console.log('App initialized successfully!');
    },

    /**
     * Setup Analysis controls
     */
    setupAnalysis() {
        const openModalBtn = document.getElementById('open-analyze-modal-btn');
        const displayBtn = document.getElementById('display-analysis-btn');
        
        // Open Analyze Modal
        if (openModalBtn) {
            openModalBtn.addEventListener('click', () => {
                this.showAnalysisDateSelectionModal();
            });
        }
        
        // Display Analysis Button (existing)
        if (displayBtn) {
            displayBtn.addEventListener('click', async () => {
                const popup = document.getElementById('analysis-popup');
                const list = document.getElementById('analysis-list');
                const overlay = document.getElementById('popup-overlay');
                
                if (popup && overlay) {
                    popup.classList.add('open');
                    overlay.classList.add('active');
                    
                    try {
                        const res = await fetch('/api/analysis');
                        const files = await res.json();
                        
                        list.innerHTML = '';
                        if (files.length === 0) {
                            list.innerHTML = '<p style="opacity: 0.7;">No analyses saved yet.</p>';
                        } else {
                            files.forEach(f => {
                                const item = document.createElement('div');
                                item.className = 'analysis-item';
                                item.style.cssText = 'padding: 10px; margin: 5px 0; background: var(--bg-secondary); border-radius: 8px; cursor: pointer;';
                                item.innerHTML = `<strong>${f.filename}</strong>`;
                                item.addEventListener('click', async () => {
                                    const content = await fetch(`/api/analysis/${f.filename}`).then(r => r.text());
                                    this.showAnalysisContent(content, f.filename);
                                });
                                list.appendChild(item);
                            });
                        }
                    } catch (e) {
                        list.innerHTML = '<p>Error loading analyses.</p>';
                    }
                    
                    document.getElementById('close-analysis-popup')?.addEventListener('click', () => {
                        popup.classList.remove('open');
                        overlay.classList.remove('active');
                    });
                }
            });
        }
    },

    /**
     * Show Date Selection Modal for Analysis
     */
    showAnalysisDateSelectionModal() {
        const overlay = document.createElement('div');
        overlay.className = 'custom-alert-overlay active';
        overlay.id = 'date-selection-overlay';
        
        const currentYear = new Date().getFullYear();
        let yearOptions = '';
        for (let y = currentYear; y >= currentYear - 5; y--) {
            yearOptions += `<option value="${y}">${y}</option>`;
        }
        
        const monthOptions = `
            <option value="1">January</option><option value="2">February</option><option value="3">March</option>
            <option value="4">April</option><option value="5">May</option><option value="6">June</option>
            <option value="7">July</option><option value="8">August</option><option value="9">September</option>
            <option value="10">October</option><option value="11">November</option><option value="12">December</option>
        `;

        overlay.innerHTML = `
            <div class="custom-alert-box" style="width: 400px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 20px;">🧠</div>
                <h3 style="margin-bottom: 20px;">Run Psychoanalysis?</h3>
                
                <div style="display: flex; gap: 10px; margin-bottom: 15px; justify-content: center;">
                    <div style="text-align: left;">
                        <label style="font-size: 0.8rem; opacity: 0.7;">Start</label><br>
                        <select id="start-month" class="date-input" style="width: 100px;">${monthOptions}</select>
                        <select id="start-year" class="date-input" style="width: 80px;">${yearOptions}</select>
                    </div>
                </div>
                
                <div style="display: flex; gap: 10px; margin-bottom: 20px; justify-content: center;">
                     <div style="text-align: left;">
                        <label style="font-size: 0.8rem; opacity: 0.7;">End</label><br>
                        <select id="end-month" class="date-input" style="width: 100px;">${monthOptions}</select>
                        <select id="end-year" class="date-input" style="width: 80px;">${yearOptions}</select>
                    </div>
                </div>

                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button id="cancel-analysis-btn" class="secondary-btn">Cancel</button>
                    <button id="confirm-analysis-btn" class="primary-btn">Yes, Analyze</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        
        // Defaults
        overlay.querySelector('#start-month').value = 1;
        overlay.querySelector('#end-month').value = 12;

        // Cancel
        document.getElementById('cancel-analysis-btn').addEventListener('click', () => {
            document.body.removeChild(overlay);
        });

        // Confirm
        document.getElementById('confirm-analysis-btn').addEventListener('click', async () => {
            const sM = document.getElementById('start-month').value;
            const sY = document.getElementById('start-year').value;
            const eM = document.getElementById('end-month').value;
            const eY = document.getElementById('end-year').value;
            
            // Construct dates (start of start-month, end of end-month logic handled by backend? 
            // Actually backend expects start_date/end_date YYYY-MM-DD. We should construct it here.)
            
            const startDate = `${sY}-${sM.padStart(2, '0')}-01`;
            
            // For end date, we need last day of month
            const lastDay = new Date(eY, eM, 0).getDate();
            const endDate = `${eY}-${eM.padStart(2, '0')}-${lastDay}`;
            
            // Show Loading
            document.body.removeChild(overlay);
            this.showAnalysisLoadingAndFetch(startDate, endDate);
        });
    },

    /**
     * Show loading GIF and fetch analysis
     */
    async showAnalysisLoadingAndFetch(start, end) {
        // Show Busy Alert/GIF
        const overlay = document.createElement('div');
        overlay.className = 'custom-alert-overlay active';
        overlay.id = 'busy-overlay';
        overlay.innerHTML = `
            <div class="custom-alert-box" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 20px;">🤖</div>
                <h3>Rona is busy...</h3>
                <p>Analyzing period ${start} to ${end}...</p>
                <img src="assets/loading.gif" alt="Loading..." style="max-width: 100px; margin: 20px auto; display: block;">
            </div>
        `;
        document.body.appendChild(overlay);
        
        try {
            const response = await fetch('/api/analyze_preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_date: start, end_date: end })
            });
            
            const result = await response.json();
            
            if (document.body.contains(overlay)) {
                document.body.removeChild(overlay);
            }
            
            if (result.ok) {
                 this.showAnalysisPreview(result.content, result.start_date, result.end_date, result.entry_count);
            } else {
                 Toast.error(result.error || 'Analysis failed');
            }
        } catch (error) {
            if (document.body.contains(overlay)) {
                document.body.removeChild(overlay);
            }
            Toast.error('Network error during analysis.');
            console.error(error);
        }
    },
    
    /**
     * Show analysis preview with save option
     */
    showAnalysisPreview(content, startDate, endDate, entryCount) {
        const overlay = document.createElement('div');
        overlay.className = 'custom-alert-overlay active';
        overlay.id = 'preview-overlay';
        overlay.innerHTML = `
            <div class="custom-alert-box" style="max-width: 600px; max-height: 80vh; overflow-y: auto;">
                <h3 style="margin-bottom: 10px;">📊 Analysis: ${startDate} to ${endDate}</h3>
                <p style="opacity: 0.7; font-size: 0.9rem; margin-bottom: 15px;">${entryCount} journal entries analyzed</p>
                <div style="background: var(--bg-secondary); padding: 15px; border-radius: 8px; white-space: pre-wrap; font-size: 0.95rem; max-height: 300px; overflow-y: auto; margin-bottom: 20px;">${content}</div>
                <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                    <input type="text" id="save-analysis-title" placeholder="Enter title to save..." class="journal-input" style="flex: 1; min-width: 150px;">
                    <button id="save-analysis-btn" class="primary-btn">💾 Save</button>
                    <button id="close-preview-btn" class="secondary-btn">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        
        // Close button
        document.getElementById('close-preview-btn').addEventListener('click', () => {
            document.body.removeChild(overlay);
        });
        
        // Save button
        document.getElementById('save-analysis-btn').addEventListener('click', async () => {
            const title = document.getElementById('save-analysis-title').value.trim();
            if (!title) {
                Toast.warning('Please enter a title to save.');
                return;
            }
            
            try {
                const res = await fetch('/api/analyze_save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: content, title: title })
                });
                const result = await res.json();
                
                if (result.ok) {
                    Toast.success(`Saved as ${result.filename}`);
                    document.body.removeChild(overlay);
                } else {
                    Toast.error(result.error || 'Save failed');
                }
            } catch (e) {
                Toast.error('Network error saving analysis.');
            }
        });
    },
    
    async fetchAndShowtitAnalysis(filename) {
         try {
             const res = await fetch(`/api/analysis/${filename}`);
             const data = await res.json();
             if (data.ok) {
                 this.showAnalysisContent(data.content, filename);
             } else {
                 Toast.error(data.error);
             }
         } catch (e) {
             Toast.error("Failed to fetch analysis");
         }
    },

    showAnalysisContent(content, title) {
        const overlay = document.createElement('div');
        overlay.className = 'modal';
        overlay.innerHTML = `
            <div class="modal-content" style="max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <h2 style="margin-bottom: 20px;">${title}</h2>
                <div style="white-space: pre-wrap; font-size: 1.1rem; line-height: 1.6;">${content}</div>
                <button class="secondary-btn" style="margin-top: 25px;">Close</button>
            </div>
        `;
        overlay.querySelector('button').addEventListener('click', () => overlay.remove());
        document.body.appendChild(overlay);
    },

    /**
     * Setup side menu toggle
     */
    setupSideMenu() {
        const menuBtn = document.getElementById('menu-button');
        const sideMenu = document.getElementById('side-menu');

        menuBtn.addEventListener('click', () => {
            menuBtn.classList.toggle('active');
            sideMenu.classList.toggle('open');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!sideMenu.contains(e.target) && !menuBtn.contains(e.target)) {
                menuBtn.classList.remove('active');
                sideMenu.classList.remove('open');
            }
        });
    },

    /**
     * Setup Privacy Mode Toggle
     */
    setupPrivacyToggle() {
        const toggle = document.getElementById('privacy-toggle');
        const statusLabel = document.getElementById('privacy-status');

        if (!toggle) return;

        // Load saved state
        const isPrivacyActive = localStorage.getItem('privacyMode') === 'true';
        toggle.checked = isPrivacyActive;
        if (isPrivacyActive) {
            document.body.classList.add('privacy-active');
            statusLabel.textContent = 'Active';
            statusLabel.style.color = 'var(--primary-color)';
            statusLabel.style.fontWeight = 'bold';
        }

        toggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('privacy-active');
                statusLabel.textContent = 'Active';
                statusLabel.style.color = 'var(--primary-color)';
                statusLabel.style.fontWeight = 'bold';
                localStorage.setItem('privacyMode', 'true');
                Toast.info('Privacy Mode Activated 👁️');
            } else {
                document.body.classList.remove('privacy-active');
                statusLabel.textContent = 'Hidden';
                statusLabel.style.color = 'var(--text-muted)';
                statusLabel.style.fontWeight = 'normal';
                localStorage.setItem('privacyMode', 'false');
                Toast.info('Privacy Mode Deactivated');
            }
        });
    },

    // ... (Navigation can stay same)

    /**
     * Setup Random Day (Flashback) functionality
     */
    setupRandomDay() {
        const btn = document.getElementById('random-day-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            // Simple click animation (Pulse)
            btn.style.transform = 'scale(0.8)';
            setTimeout(() => {
                btn.style.transform = 'scale(1)';
            }, 100);
            
            try {
                const res = await fetch('/api/analysis/random');
                const data = await res.json();
                
                if (data.ok) {
                    this.showAnalysisContent(data.content, data.filename);
                } else {
                    Toast.info("The user still does't use the analyis button yet and the folder empty"); 
                }
            } catch (e) {
                 Toast.error("Failed to fetch random analysis");
            }
        });
    },

    /**
     * Setup quick notes functionality
     */
    setupQuickNotes() {
        const btn = document.getElementById('quick-notes-btn');
        const panel = document.getElementById('quick-notes-panel');
        const textarea = document.getElementById('quick-notes-textarea');

        // Load saved notes from Storage
        textarea.value = Storage.getQuickNotes() || '';

        // Toggle panel
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            panel.classList.toggle('open');
        });

        // Auto-save on input with debounce
        let saveTimeout;
        textarea.addEventListener('input', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                Storage.setQuickNotes(textarea.value);
            }, 500);
        });

        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            if (!panel.contains(e.target) && !btn.contains(e.target)) {
                panel.classList.remove('open');
            }
        });
    },

    /**
     * Setup search and favorites functionality
     */
    setupSearch() {
        const searchInput = document.getElementById('journal-search');
        const searchBtn = document.getElementById('search-btn');
        const clearBtn = document.getElementById('clear-search-btn');
        const favoritesBtn = document.getElementById('show-favorites-btn');
        const showAllBtn = document.getElementById('show-all-btn');

        if (!searchInput) return;

        // Search function
        const performSearch = () => {
            const query = searchInput.value.toLowerCase().trim();
            if (query) {
                Journal.renderJournals(query);
            } else {
                Journal.renderJournals();
            }
        };

        searchBtn?.addEventListener('click', performSearch);
        searchInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') performSearch();
        });

        // Clear search button
        clearBtn?.addEventListener('click', () => {
            searchInput.value = '';
            Journal.renderJournals();
        });

        // Clear search on input clear
        searchInput.addEventListener('input', () => {
            if (searchInput.value === '') {
                Journal.renderJournals();
            }
        });

        // Favorites filter
        favoritesBtn?.addEventListener('click', () => {
            Journal.renderJournals(null, true);
        });

        // Show All button
        showAllBtn?.addEventListener('click', () => {
            searchInput.value = '';
            Journal.renderJournals();
        });
    },

    /**
     * Setup backup and restore functionality
     */
    setupBackup() {
        // Export backup button
        const exportBtn = document.getElementById('export-backup-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                const data = Storage.exportAllData();
                const blob = new Blob([data], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `daily-productivity-backup-${new Date().toISOString().split('T')[0]}.json`;
                link.click();
                URL.revokeObjectURL(url);
                Toast.success('Backup exported successfully!');
            });
        }

        // Import backup button
        const importBtn = document.getElementById('import-backup-btn');
        const fileInput = document.getElementById('import-file-input');

        if (importBtn && fileInput) {
            importBtn.addEventListener('click', () => fileInput.click());

            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = async (event) => {
                    const result = await Storage.importData(event.target.result);
                    if (result.success) {
                        Toast.show(result.message, 'success');
                        setTimeout(() => location.reload(), 1500);
                    } else {
                        Toast.show(result.message, 'error');
                    }
                };
                reader.readAsText(file);
                fileInput.value = '';
            });
        }
    },

    /**
     * Setup custom resize handles for large textareas
     */
    setupResizableTextareas() {
        // Target textareas that should have custom resize handles
        const textareaIds = [
            'journal-details',
            'template-input',
            'week-details',
            'monthly-details'
        ];

        textareaIds.forEach(id => {
            const textarea = document.getElementById(id);
            if (!textarea) return;

            // Wrap textarea if not already wrapped
            if (!textarea.parentElement.classList.contains('resizable-textarea')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'resizable-textarea';
                textarea.parentNode.insertBefore(wrapper, textarea);
                wrapper.appendChild(textarea);
            }

            const wrapper = textarea.parentElement;

            // Create resize handle
            const handle = document.createElement('div');
            handle.className = 'resize-handle';
            handle.title = 'Drag to resize';
            wrapper.appendChild(handle);

            // Resize functionality
            let isResizing = false;
            let startY = 0;
            let startHeight = 0;

            handle.addEventListener('mousedown', (e) => {
                isResizing = true;
                startY = e.clientY;
                startHeight = textarea.offsetHeight;
                document.body.style.cursor = 'ns-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });

            document.addEventListener('mousemove', (e) => {
                if (!isResizing) return;
                const delta = e.clientY - startY;
                const newHeight = Math.max(100, startHeight + delta);
                textarea.style.height = newHeight + 'px';
            });

            document.addEventListener('mouseup', () => {
                if (isResizing) {
                    isResizing = false;
                    document.body.style.cursor = '';
                    document.body.style.userSelect = '';
                }
            });

            // Touch support for mobile
            handle.addEventListener('touchstart', (e) => {
                isResizing = true;
                startY = e.touches[0].clientY;
                startHeight = textarea.offsetHeight;
                e.preventDefault();
            }, { passive: false });

            document.addEventListener('touchmove', (e) => {
                if (!isResizing) return;
                const delta = e.touches[0].clientY - startY;
                const newHeight = Math.max(100, startHeight + delta);
                textarea.style.height = newHeight + 'px';
            }, { passive: true });

            document.addEventListener('touchend', () => {
                isResizing = false;
            });
        });
    }
};

/**
 * Custom Alert Utility - Styled modals instead of browser alerts
 */
const CustomAlert = {
    overlay: null,
    confirmCallback: null,

    init() {
        this.overlay = document.getElementById('custom-alert-overlay');
        if (!this.overlay) return;

        const confirmBtn = document.getElementById('custom-alert-confirm');
        const cancelBtn = document.getElementById('custom-alert-cancel');

        if (!confirmBtn || !cancelBtn) return;

        confirmBtn.addEventListener('click', () => {
            const callback = this.confirmCallback;
            this.close();
            if (callback) callback(true);
        });

        cancelBtn.addEventListener('click', () => {
            const callback = this.confirmCallback;
            this.close();
            if (callback) callback(false);
        });

        // Close on overlay click
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                const callback = this.confirmCallback;
                this.close();
                if (callback) callback(false);
            }
        });
    },

    show(options) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            question: '❓'
        };

        document.getElementById('custom-alert-icon').textContent = icons[options.type] || icons.info;
        document.getElementById('custom-alert-title').textContent = options.title || 'Alert';
        document.getElementById('custom-alert-message').textContent = options.message || '';

        const cancelBtn = document.getElementById('custom-alert-cancel');
        const confirmBtn = document.getElementById('custom-alert-confirm');

        if (options.showCancel) {
            cancelBtn.style.display = 'block';
            cancelBtn.textContent = options.cancelText || 'Cancel';
        } else {
            cancelBtn.style.display = 'none';
        }

        confirmBtn.textContent = options.confirmText || 'OK';

        this.confirmCallback = options.onConfirm || null;
        this.overlay.classList.add('active');
    },

    close() {
        this.overlay.classList.remove('active');
        this.confirmCallback = null;
    },

    // Helper methods
    success(message, title = 'Success') {
        this.show({ type: 'success', title, message });
    },

    error(message, title = 'Error') {
        this.show({ type: 'error', title, message });
    },

    warning(message, title = 'Warning') {
        this.show({ type: 'warning', title, message });
    },

    info(message, title = 'Info') {
        this.show({ type: 'info', title, message });
    },

    confirm(message, onConfirm, title = 'Confirm') {
        this.show({
            type: 'question',
            title,
            message,
            showCancel: true,
            cancelText: 'Cancel',
            confirmText: 'Yes',
            onConfirm
        });
    }
};

// Backward compatibility - Toast redirects to CustomAlert
const Toast = {
    success(message) { CustomAlert.success(message); },
    error(message) { CustomAlert.error(message); },
    warning(message) { CustomAlert.warning(message); },
    info(message) { CustomAlert.info(message); }
};

window.CustomAlert = CustomAlert;
window.Toast = Toast;

// Start the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
    CustomAlert.init();
});

