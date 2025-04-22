let timerElement = null;
let intervalId = null;
let seconds = 0;

// Start timer when loading spinner is visible
const observer = new MutationObserver(() => {
    const spinner = document.querySelector(".dash-loading");
    const image = document.getElementById("signal-plot");

    if (spinner && spinner.classList.contains("dash-loading-show")) {
        timerElement = document.getElementById("time-counter");
        if (!intervalId && timerElement) {
            seconds = 0;
            intervalId = setInterval(() => {
                seconds += 1;
                timerElement.innerText = `Time: ${seconds}s`;
            }, 1000);
        }
    }

    // Stop timer when the image is fully loaded
    if (image && image.complete && intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
});

observer.observe(document.body, { childList: true, subtree: true });
