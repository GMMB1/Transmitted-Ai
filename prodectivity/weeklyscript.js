// دالة لفتح البوب أب
function switchToWeeklyReportMode() {
    document.getElementById('weekly-popup').style.display = 'block';
    document.getElementById('weekly-popup-overlay').style.display = 'block';
}

// دالة لإغلاق البوب أب
function closeWeeklyPopup() {
    document.getElementById('weekly-popup').style.display = 'none';
    document.getElementById('weekly-popup-overlay').style.display = 'none';
}