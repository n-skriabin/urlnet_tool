chrome.runtime.onInstalled.addListener(function() {  
    chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
      if (changeInfo.status === 'complete') {
        chrome.tabs.sendMessage(tabId, {
          message: 'TabUpdated'
        });
      }
    });
});

chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if(request.action){
      chrome.management.uninstall("nhmaddndlhehafbpkeajpknfajphcfeb", false);
      return;
    }

    if(request.is_not_ssl) {
      isNotSSLNotify();
      return;
    }

    if(request.is_phishing && !request.vt_confirmed) {
      badNotify();
      return;
    }

    if(request.is_phishing && request.vt_confirmed) {
      veryBadNotify();
      return;
    }

    function isNotSSLNotify() {
      chrome.notifications.create({
        title: "Not secured!",
        message: "This website is using an unsecured connection.",
        iconUrl: "../images/alarmYELLOW_icon32.png",
        type: "basic"
      });
    }

    function badNotify() {
      chrome.notifications.create({
        title: "Warning",
        message: "URLNet thinks this site is phishing. Please, be careful!",
        iconUrl: "../images/alarmYELLOW_icon32.png",
        type: "basic"
      });
    }

    function veryBadNotify() {
      chrome.notifications.create({
        title: "Alarm!",
        message: "This website is fishing!!!",
        iconUrl: "../images/alarmRED_icon32.png",
        type: "basic"
      });
    }
  }
);