/**
 * Themes Module - Handles theme switching and neon effects
 */

const Themes = {
    themeConfigs: {
        default: {
            primary: '#ff3232',
            secondary: '#3c55e4',
            backPrimary: '#1a0a0a',
            backSecondary: '#0a0a1a',
            container: '#181820',
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

    currentNeon: 'none',
    isPaletteOpen: false,
    currentPattern: 'none',

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

        // Update active dot state
        document.querySelectorAll('.pattern-dot').forEach(dot => {
            dot.classList.remove('active');
            if (dot.dataset.pattern === patternName) {
                dot.classList.add('active');
            }
        });
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

        this.apply(savedTheme);
        this.applyNeon(savedNeon);
        this.applyPattern(savedPattern);

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
