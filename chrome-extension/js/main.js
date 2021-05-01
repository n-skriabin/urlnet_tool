document.addEventListener('DOMContentLoaded', function() {
    var aiElement = document.getElementById('enabled');

    aiElement.addEventListener('change', function() {
        chrome.tabs.query({currentWindow: true, active: true}, function (tabs) {
            chrome.runtime.sendMessage(
                { 
                    action: 'isEnabledAction',
                    enabled: false
                });
        });
    });
});