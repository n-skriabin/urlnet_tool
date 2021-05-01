chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.message === 'TabUpdated') {
      const url = new URL(document.location.href);
      if(url.protocol != 'https:') {
        chrome.runtime.sendMessage(
        { 
          is_not_ssl: true
        });
      }

      var xhrUrlNet = new XMLHttpRequest();
      xhrUrlNet.open("GET", "http://127.0.0.1:5000/check-url/" + document.location.hostname, true);
      xhrUrlNet.setRequestHeader("Access-Control-Allow-Origin", "*");
      xhrUrlNet.onreadystatechange = checkUrl_Callback;
      xhrUrlNet.send();

      function checkUrl_Callback() {
        if (xhrUrlNet.readyState === XMLHttpRequest.DONE) {
          if (xhrUrlNet.status === 200) {
            resultUrlNet = JSON.parse(xhrUrlNet.responseText);
            if(resultUrlNet.is_phishing) {
              var xhr_vt = new XMLHttpRequest();
              xhr_vt.open("GET", "http://127.0.0.1:5000/check-url-vt/" + document.location.hostname, true);
              xhr_vt.setRequestHeader("Access-Control-Allow-Origin", "*");
              xhr_vt.onreadystatechange = vt_Callback;
              xhr_vt.send();

              function vt_Callback() {
                if (xhr_vt.readyState === XMLHttpRequest.DONE) {
                  if (xhr_vt.status === 200) {
                    result_vt = JSON.parse(xhr_vt.responseText);
                    chrome.runtime.sendMessage(
                      { 
                        is_phishing: true, 
                        vt_confirmed: result_vt.vt_confirmed
                      });
                  }
                }
              }
            }
            else 
            {
              chrome.runtime.sendMessage(
                { 
                  is_phishing: false, 
                  vt_confirmed: false 
                });
            }
          }
        }
      };
    }
});
