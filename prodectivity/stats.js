function showStatistics() {
    const startDate = new Date(document.getElementById("start-date").value);
    const endDate = new Date(document.getElementById("end-date").value);
    const statsUnit = document.getElementById("stats-unit").value;

    if (!startDate || !endDate || isNaN(startDate) || isNaN(endDate)) {
        alert("Please specify a valid date range!");
        return;
    }

    // استرجاع البيانات من LocalStorage
    const Journals = JSON.parse(localStorage.getItem("Journals")) || [];
    const filteredJournals = Journals.filter(Journal => {
        const JournalDate = new Date(Journal.date);
        return JournalDate >= startDate && JournalDate <= endDate;
    });

    // إعداد البيانات للرسم البياني
    let dates = [];
    let ratings = [];

    if (statsUnit === "daily") {
        let currentDate = new Date(startDate);
        while (currentDate <= endDate) {
            const Journal = filteredJournals.find(Journal =>
                new Date(Journal.date).toDateString() === currentDate.toDateString()
            );

            dates.push(currentDate.toDateString());
            ratings.push(Journal ? Journal.dailyRating : -1); // إذا لم توجد بيانات، ضع -1
            currentDate.setDate(currentDate.getDate() + 1);
        }
    }

    renderChart(dates, ratings);
}
function renderChart(dates, ratings) {
    const ctx = document.getElementById("statsChart").getContext("2d");

    if (window.chartInstance) {
        window.chartInstance.destroy();
    }

    // تحسين التدرج اللوني مع شفافية أقل في الأسفل
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, "rgba(255, 50, 50, 0.3)");
    gradient.addColorStop(1, "rgba(255, 50, 50, 0.1)"); // زيادة التعتيم في الأسفل

    window.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates,
            datasets: [{
                label: "Daily Ratings",
                data: ratings,
                borderColor: "var(--primary-color)",
                borderWidth: 3,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                spanGaps: false,
                pointRadius: 6,
                pointHoverRadius: 10,
                pointBackgroundColor: "var(--primary-color)",
                pointBorderColor: "white",
                pointBorderWidth: 2,
                pointHoverBackgroundColor: "var(--secondary-color)",
                pointHoverBorderColor: "white",
                pointHoverBorderWidth: 3,
                hitRadius: 60,  // التأكد من تحسين مكان التفاعل حول النقطة
                custom: function (chart) {
                    const points = chart.getElementsAtEventForMode(chart.event, 'nearest', {intersect: true}, false);
                    return points.length ? points[0]._model : null;
                }
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',        // تغيير من 'nearest' إلى 'index'
                intersect: true,      // تغيير إلى true للتفاعل المباشر مع النقاط
                axis: 'xy'           // تغيير من 'x' إلى 'xy' للتفاعل في كلا المحورين
            },
            plugins: {
                legend: {
                    position: "top",
                    align: "center",
                    labels: {
                        color: "white",
                        font: {
                            family: "'Cairo', sans-serif",
                            size: 14,
                            weight: 'bold'
                        },
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: "var(--container-color)",
                    titleColor: "var(--primary-color)",
                    bodyColor: "white",
                    borderColor: "var(--primary-color)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: {
                        family: "'Cairo', sans-serif",
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        family: "'Cairo', sans-serif",
                        size: 13
                    },
                    displayColors: false,
                    position: 'nearest',
                    caretSize: 6,
                    caretPadding: 2
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 10,
                    grid: {
                        color: "rgba(255, 255, 255, 0.1)",
                        drawBorder: false
                    },
                    ticks: {
                        color: "white",
                        font: {
                            family: "'Cairo', sans-serif",
                            size: 12
                        },
                        padding: 10,
                        stepSize: 1
                    }
                },
                x: {
                    grid: {
                        color: "rgba(255, 255, 255, 0.1)",
                        drawBorder: false
                    },
                    ticks: {
                        color: "white",
                        font: {
                            family: "'Cairo', sans-serif",
                            size: 12
                        },
                        padding: 10,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            },
            hover: {
                mode: 'index',        // تحسين وضع التحويم
                intersect: true,      // تفعيل التقاطع المباشر
                axis: 'xy'           // التفاعل في كلا المحورين
            }
        }
    });
}
