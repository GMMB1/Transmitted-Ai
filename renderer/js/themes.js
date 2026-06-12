/**
 * Themes Module - Handles theme switching and neon effects
 */

const Themes = {
    themeConfigs: {
        // v10 native look — purple/cyan on deep dark
        default: {
            primary: '#7c3aed',
            secondary: '#06b6d4',
            backPrimary: '#0a0a0f',
            backSecondary: '#12121a',
            container: '#1a1a28',
            cardBg: '#1e1e2a'
        },
        blue: {
            primary: '#00d4ff',
            secondary: '#0066ff',
            backPrimary: '#001428',
            backSecondary: '#002850',
            container: '#0a1a2f',
            cardBg: '#0d2040'
        },
        pink: {
            primary: '#ff6b9d',
            secondary: '#c44569',
            backPrimary: '#1a0a14',
            backSecondary: '#280a1e',
            container: '#201018',
            cardBg: '#2a1020'
        },
        purple: {
            primary: '#a855f7',
            secondary: '#6366f1',
            backPrimary: '#140a28',
            backSecondary: '#0a0a1e',
            container: '#1a1428',
            cardBg: '#201830'
        },
        dark: {
            primary: '#6b7280',
            secondary: '#4b5563',
            backPrimary: '#0f0f0f',
            backSecondary: '#1a1a1a',
            container: '#141414',
            cardBg: '#1a1a1a',
            textColor: '#ffffff',
            textMuted: '#8a8a9a',
            borderColor: 'rgba(255, 255, 255, 0.1)'
        },
        // Light Themes
        cream: {
            primary: '#d97706',
            secondary: '#b45309',
            backPrimary: '#fef3e2',
            backSecondary: '#fde8cd',
            container: '#ffffff',
            cardBg: '#fff8f0',
            textColor: '#1f1f1f',
            textMuted: '#6b6b6b',
            borderColor: 'rgba(0, 0, 0, 0.1)',
            isLight: true
        },
        sky: {
            primary: '#0891b2',
            secondary: '#0e7490',
            backPrimary: '#e0f7fa',
            backSecondary: '#b2ebf2',
            container: '#ffffff',
            cardBg: '#f0feff',
            textColor: '#1a1a1a',
            textMuted: '#5a5a5a',
            borderColor: 'rgba(0, 0, 0, 0.1)',
            isLight: true
        }
    },

    // Neon glow colors
    neonColors: {
        none: { glow: 'transparent', glowRgb: '0, 0, 0' },
        default: { glow: '#ff3232', glowRgb: '255, 50, 50' },
        blue: { glow: '#00d4ff', glowRgb: '0, 212, 255' },
        pink: { glow: '#ff6b9d', glowRgb: '255, 107, 157' },
        purple: { glow: '#a855f7', glowRgb: '168, 85, 247' },
        // Light theme neons
        amber: { glow: '#f59e0b', glowRgb: '245, 158, 11' },
        teal: { glow: '#14b8a6', glowRgb: '20, 184, 166' }
    },

    // Pattern ink colors — independent of theme and pattern shape
    patternInks: {
        red:    '#ff3232',
        blue:   '#00d4ff',
        pink:   '#ff6b9d',
        purple: '#7c3aed',
        amber:  '#f59e0b',
        teal:   '#14b8a6',
        gray:   '#9ca3af'
    },

    currentNeon: 'none',
    isPaletteOpen: false,
    currentPattern: 'none',
    currentPatternColor: 'purple',

    /**
     * Apply a theme
     */
    apply(themeName) {
        const theme = this.themeConfigs[themeName];
        if (!theme) return;

        const root = document.documentElement;
        root.style.setProperty('--primary-color', theme.primary);
        root.style.setProperty('--secondary-color', theme.secondary);
        root.style.setProperty('--back-primary-color', theme.backPrimary);
        root.style.setProperty('--back-secondary-color', theme.backSecondary);
        root.style.setProperty('--container-color', theme.container);
        root.style.setProperty('--card-bg', theme.cardBg);

        // v10 design tokens — the redesigned CSS reads these, not the legacy
        // names above. Without them only text/border colors changed.
        const rgb = (hex) => {
            const h = hex.replace('#', '');
            return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
        };
        root.style.setProperty('--accent', theme.primary);
        root.style.setProperty('--accent2', theme.secondary);
        root.style.setProperty('--accent-glow', `rgba(${rgb(theme.primary)},0.3)`);
        root.style.setProperty('--accent2-glow', `rgba(${rgb(theme.secondary)},0.3)`);
        root.style.setProperty('--accent-gradient', `linear-gradient(135deg, ${theme.primary}, ${theme.secondary})`);
        root.style.setProperty('--border-active', `rgba(${rgb(theme.primary)},0.6)`);
        root.style.setProperty('--bg-base', theme.backPrimary);
        root.style.setProperty('--bg-surface', theme.backSecondary);
        root.style.setProperty('--bg-elevated', theme.container);
        if (theme.isLight) {
            root.style.setProperty('--bg-glass', 'rgba(0,0,0,0.04)');
            root.style.setProperty('--bg-glass-hover', 'rgba(0,0,0,0.08)');
        } else {
            root.style.setProperty('--bg-glass', 'rgba(255,255,255,0.04)');
            root.style.setProperty('--bg-glass-hover', 'rgba(255,255,255,0.08)');
        }

        // Light theme specific colors
        if (theme.textColor) {
            root.style.setProperty('--text-color', theme.textColor);
        } else {
            root.style.setProperty('--text-color', '#ffffff');
        }
        if (theme.textMuted) {
            root.style.setProperty('--text-muted', theme.textMuted);
        } else {
            root.style.setProperty('--text-muted', '#8a8a9a');
        }
        if (theme.borderColor) {
            root.style.setProperty('--border-color', theme.borderColor);
        } else {
            root.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.1)');
        }

        // Toggle light-theme class on body
        if (theme.isLight) {
            document.body.classList.add('light-theme');
        } else {
            document.body.classList.remove('light-theme');
        }

        // Save theme preference
        Storage.setTheme(themeName);

        // Update active dot state
        document.querySelectorAll('.palette-dot[data-theme]').forEach(dot => {
            dot.classList.remove('active');
            if (dot.dataset.theme === themeName) {
                dot.classList.add('active');
            }
        });

        // Update mobile dots
        document.querySelectorAll('.mobile-dot[data-theme]').forEach(dot => {
            dot.classList.remove('active');
            if (dot.dataset.theme === themeName) {
                dot.classList.add('active');
            }
        });
    },

    /**
     * Apply neon glow effect
     */
    applyNeon(neonName) {
        const neon = this.neonColors[neonName];
        if (!neon) return;

        this.currentNeon = neonName;
        const root = document.documentElement;

        if (neonName === 'none') {
            root.style.setProperty('--neon-glow', 'transparent');
            root.style.setProperty('--neon-glow-rgb', '0, 0, 0');
            root.style.setProperty('--neon-intensity', '0');
            document.body.classList.remove('neon-enabled');
        } else {
            root.style.setProperty('--neon-glow', neon.glow);
            root.style.setProperty('--neon-glow-rgb', neon.glowRgb);
            root.style.setProperty('--neon-intensity', '1');
            document.body.classList.add('neon-enabled');
        }

        // Save neon preference
        Storage.setNeon(neonName);

        // Update active neon dot state
        document.querySelectorAll('.neon-dot').forEach(dot => {
            dot.classList.remove('active');
            if (dot.dataset.neon === neonName) {
                dot.classList.add('active');
            }
        });
    },

    /**
     * Apply background pattern
     */
    applyPattern(patternName) {
        // Remove existing pattern classes
        const patterns = ['none', 'hex', 'wave', 'paper', 'lines', 'curves'];
        patterns.forEach(p => {
            if (p !== 'none') document.body.classList.remove(`pattern-${p}`);
        });

        this.currentPattern = patternName;

        if (patternName !== 'none') {
            document.body.classList.add(`pattern-${patternName}`);
        }

        // Save preference
        Storage.setPattern(patternName);

        this.renderPattern();

        // Update active dot state
        document.querySelectorAll('.pattern-dot').forEach(dot => {
            dot.classList.remove('active');
            if (dot.dataset.pattern === patternName) {
                dot.classList.add('active');
            }
        });
    },

    /**
     * Apply pattern ink color (independent of pattern shape)
     */
    applyPatternColor(colorName) {
        if (!this.patternInks[colorName]) return;
        this.currentPatternColor = colorName;
        Storage.setPatternColor(colorName);
        this.renderPattern();

        document.querySelectorAll('.pcolor-dot').forEach(dot => {
            dot.classList.toggle('active', dot.dataset.pcolor === colorName);
        });
    },

    /**
     * Draw the current pattern in the current ink color.
     * Patterns were previously hardcoded purple in CSS — this builds the
     * same SVG/gradient backgrounds with the user-chosen color instead.
     */
    renderPattern() {
        const overlay = document.querySelector('.pattern-overlay');
        if (!overlay) return;

        const hex = this.patternInks[this.currentPatternColor] || '#7c3aed';
        const enc = hex.replace('#', '%23');
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);

        switch (this.currentPattern) {
            case 'hex':
                overlay.style.backgroundImage = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='49' viewBox='0 0 28 49'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='${enc}' fill-opacity='0.06'%3E%3Cpath d='M13.99 9.25l13 7.5v15l-13 7.5L1 31.75v-15l12.99-7.5zM3 17.9v12.7l10.99 6.34 11-6.35V17.9l-11-6.34L3 17.9zM0 15l12.98-7.5V0h-2v6.35L0 12.69v2.3zm0 18.5L12.98 41v8h-2v-6.85L0 35.81v-2.3zM15 0v7.5L27.99 15H28v-2.31h-.01L17 6.35V0h-2zm0 49v-8l12.99-7.5H28v2.31h-.01L17 42.15V49h-2z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`;
                overlay.style.backgroundSize = '';
                overlay.style.opacity = '1';
                break;
            case 'lines':
                overlay.style.backgroundImage =
                    `linear-gradient(rgba(${r},${g},${b},0.05) 1px, transparent 1px),` +
                    `linear-gradient(90deg, rgba(${r},${g},${b},0.05) 1px, transparent 1px)`;
                overlay.style.backgroundSize = '40px 40px';
                overlay.style.opacity = '1';
                break;
            case 'curves':
                overlay.style.backgroundImage = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='50' height='50' viewBox='0 0 50 50'%3E%3Crect x='2' y='2' width='46' height='46' fill='none' stroke='${enc}' stroke-opacity='0.07' stroke-width='1' rx='0' ry='0'/%3E%3Cpath fill='none' stroke='${enc}' stroke-opacity='0.05' d='M2 2 Q 25 10, 48 2 M48 2 Q 40 25, 48 48 M48 48 Q 25 40, 2 48 M2 48 Q 10 25, 2 2'/%3E%3C/svg%3E")`;
                overlay.style.backgroundSize = '';
                overlay.style.opacity = '1';
                break;
            default:
                // 'paper' (color-independent noise) and 'none' fall back to CSS
                overlay.style.backgroundImage = '';
                overlay.style.backgroundSize = '';
                overlay.style.opacity = '';
                break;
        }
    },

    /**
     * Toggle palette open/close
     */
    togglePalette() {
        this.isPaletteOpen = !this.isPaletteOpen;
        const palette = document.getElementById('theme-palette');
        const toggleBtn = document.getElementById('theme-toggle-btn');

        if (this.isPaletteOpen) {
            palette.classList.add('open');
            toggleBtn.classList.add('active');
        } else {
            palette.classList.remove('open');
            toggleBtn.classList.remove('active');
        }
    },

    /**
     * Close palette
     */
    closePalette() {
        this.isPaletteOpen = false;
        const palette = document.getElementById('theme-palette');
        const toggleBtn = document.getElementById('theme-toggle-btn');
        if (palette) palette.classList.remove('open');
        if (toggleBtn) toggleBtn.classList.remove('active');
    },

    /**
     * Initialize theme from storage
     */
    init() {
        const savedTheme = Storage.getTheme() || 'default';
        const savedNeon = Storage.getNeon() || 'none';
        const savedPattern = Storage.getPattern() || 'none';
        const savedPatternColor = Storage.getPatternColor() || 'purple';

        this.apply(savedTheme);
        this.applyNeon(savedNeon);
        this.currentPatternColor = savedPatternColor;
        this.applyPattern(savedPattern);
        this.applyPatternColor(savedPatternColor);

        // Toggle button click
        const toggleBtn = document.getElementById('theme-toggle-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.togglePalette();
            });
        }

        // Theme dots click
        document.querySelectorAll('.palette-dot[data-theme]').forEach(dot => {
            dot.addEventListener('click', (e) => {
                e.stopPropagation();
                const theme = dot.getAttribute('data-theme');
                this.apply(theme);
            });
        });

        // Neon dots click
        document.querySelectorAll('.neon-dot').forEach(dot => {
            dot.addEventListener('click', (e) => {
                e.stopPropagation();
                const neon = dot.getAttribute('data-neon');
                this.applyNeon(neon);
            });
        });

        // Pattern dots click
        document.querySelectorAll('.pattern-dot').forEach(dot => {
            dot.addEventListener('click', (e) => {
                e.stopPropagation();
                const pattern = dot.getAttribute('data-pattern');
                this.applyPattern(pattern);
            });
        });

        // Pattern ink color dots click
        document.querySelectorAll('.pcolor-dot').forEach(dot => {
            dot.addEventListener('click', (e) => {
                e.stopPropagation();
                this.applyPatternColor(dot.getAttribute('data-pcolor'));
            });
        });

        // Close palette when clicking outside
        document.addEventListener('click', (e) => {
            const container = document.querySelector('.theme-widget-container');
            if (container && !container.contains(e.target)) {
                this.closePalette();
            }
        });

        // Mobile Theme Dots
        document.querySelectorAll('.mobile-dot[data-theme]').forEach(dot => {
            dot.addEventListener('click', () => {
                const theme = dot.getAttribute('data-theme');
                this.apply(theme);
                // Update mobile active state
                document.querySelectorAll('.mobile-dot[data-theme]').forEach(d => d.classList.remove('active'));
                dot.classList.add('active');
            });
        });

        // Mobile Neon Dots
        document.querySelectorAll('.mobile-dot[data-neon]').forEach(dot => {
            dot.addEventListener('click', () => {
                const neon = dot.getAttribute('data-neon');
                this.applyNeon(neon);
                // Update mobile active state
                document.querySelectorAll('.mobile-dot[data-neon]').forEach(d => d.classList.remove('active'));
                dot.classList.add('active');
            });
        });

        // Set initial active states for mobile
        const savedMobileThemeDot = document.querySelector(`.mobile-dot[data-theme="${savedTheme}"]`);
        if (savedMobileThemeDot) savedMobileThemeDot.classList.add('active');

        const savedMobileNeonDot = document.querySelector(`.mobile-dot[data-neon="${savedNeon}"]`);
        if (savedMobileNeonDot) savedMobileNeonDot.classList.add('active');
    }
};

window.Themes = Themes;
