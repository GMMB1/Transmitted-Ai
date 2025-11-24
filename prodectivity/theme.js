function toggleTheme(theme) {
    if (theme === 'default') {
        document.documentElement.style.setProperty('--primary-color', '#ff3232'); // Red
        document.documentElement.style.setProperty('--secondary-color', '#4f47ba'); // Purple
        document.documentElement.style.setProperty('--back-primary-color', '#3a0000');
        document.documentElement.style.setProperty('--back-secondary-color', '#1e132c');
        document.documentElement.style.setProperty('--back-center', '#000000');
        document.documentElement.style.setProperty('--container-color', '#151515');
    } else if (theme === 'blue') {
    // الحصول على القيم الفعلية للمتغيرات
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--b-primary-color').trim();
    const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--b-secondary-color').trim();
    const backPrimaryColor = getComputedStyle(document.documentElement).getPropertyValue('--b-back-primary-color').trim();
    const backSecondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--b-back-secondary-color').trim();
    const backCenter = getComputedStyle(document.documentElement).getPropertyValue('--b-back-center').trim();
    const containerColor = getComputedStyle(document.documentElement).getPropertyValue('--b-container-color').trim();

    // تعيين هذه القيم كقيمة للمتغيرات الجديدة
    document.documentElement.style.setProperty('--primary-color', primaryColor);
    document.documentElement.style.setProperty('--secondary-color', secondaryColor);
    document.documentElement.style.setProperty('--back-primary-color', backPrimaryColor);
    document.documentElement.style.setProperty('--back-secondary-color', backSecondaryColor);
    document.documentElement.style.setProperty('--back-center', backCenter);
    document.documentElement.style.setProperty('--container-color', containerColor);
} else if (theme === 'pink') {
    // الحصول على القيم الفعلية للمتغيرات
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--pk-primary-color').trim();
    const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--pk-secondary-color').trim();
    const backPrimaryColor = getComputedStyle(document.documentElement).getPropertyValue('--pk-back-primary-color').trim();
    const backSecondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--pk-back-secondary-color').trim();
    const backCenter = getComputedStyle(document.documentElement).getPropertyValue('--pk-back-center').trim();
    const containerColor = getComputedStyle(document.documentElement).getPropertyValue('--pk-container-color').trim();

    // تعيين هذه القيم كقيمة للمتغيرات الجديدة
    document.documentElement.style.setProperty('--primary-color', primaryColor);
    document.documentElement.style.setProperty('--secondary-color', secondaryColor);
    document.documentElement.style.setProperty('--back-primary-color', backPrimaryColor);
    document.documentElement.style.setProperty('--back-secondary-color', backSecondaryColor);
    document.documentElement.style.setProperty('--back-center', backCenter);
    document.documentElement.style.setProperty('--container-color', containerColor);
} else if (theme === 'purple') {
    // الحصول على القيم الفعلية للمتغيرات
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--p-primary-color').trim();
    const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--p-secondary-color').trim();
    const backPrimaryColor = getComputedStyle(document.documentElement).getPropertyValue('--p-back-primary-color').trim();
    const backSecondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--p-back-secondary-color').trim();
    const backCenter = getComputedStyle(document.documentElement).getPropertyValue('--p-back-center').trim();
    const containerColor = getComputedStyle(document.documentElement).getPropertyValue('--p-container-color').trim();

    // تعيين هذه القيم كقيمة للمتغيرات الجديدة
    document.documentElement.style.setProperty('--primary-color', primaryColor);
    document.documentElement.style.setProperty('--secondary-color', secondaryColor);
    document.documentElement.style.setProperty('--back-primary-color', backPrimaryColor);
    document.documentElement.style.setProperty('--back-secondary-color', backSecondaryColor);
    document.documentElement.style.setProperty('--back-center', backCenter);
    document.documentElement.style.setProperty('--container-color', containerColor);
} else if (theme === 'dark') {
    // الحصول على القيم الفعلية للمتغيرات
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--d-primary-color').trim();
    const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--d-secondary-color').trim();
    const backPrimaryColor = getComputedStyle(document.documentElement).getPropertyValue('--d-back-primary-color').trim();
    const backSecondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--d-back-secondary-color').trim();
    const backCenter = getComputedStyle(document.documentElement).getPropertyValue('--d-back-center').trim();
    const containerColor = getComputedStyle(document.documentElement).getPropertyValue('--d-container-color').trim();

    // تعيين هذه القيم كقيمة للمتغيرات الجديدة
    document.documentElement.style.setProperty('--primary-color', primaryColor);
    document.documentElement.style.setProperty('--secondary-color', secondaryColor);
    document.documentElement.style.setProperty('--back-primary-color', backPrimaryColor);
    document.documentElement.style.setProperty('--back-secondary-color', backSecondaryColor);
    document.documentElement.style.setProperty('--back-center', backCenter);
    document.documentElement.style.setProperty('--container-color', containerColor);
}
localStorage.setItem('selectedTheme', theme);
}
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('selectedTheme'); // استرجاع الثيم المحفوظ
    if (savedTheme) {
        toggleTheme(savedTheme); // تطبيق الثيم المحفوظ
    } else {
        // إذا لم يكن هناك ثيم محفوظ، يمكنك تعيين ثيم افتراضي هنا
        toggleTheme('default');
    }
});
