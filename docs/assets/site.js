// Sidebar: mark the section currently in view, and open/close on narrow screens.
(function () {
  var nav = document.querySelector('.nav');
  var btn = document.querySelector('.navbtn');

  if (btn && nav) {
    btn.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
    });
    nav.addEventListener('click', function (e) {
      if (e.target.closest('a') && window.innerWidth <= 1040) {
        nav.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  var links = {};
  nav.querySelectorAll('a.item[href^="#"]').forEach(function (a) {
    links[a.getAttribute('href').slice(1)] = a;
  });
  var sections = [].filter.call(document.querySelectorAll('section[id]'), function (s) {
    return links[s.id];
  });
  if (!sections.length) return;

  function mark() {
    // The active section is the last one whose top has passed the reading line.
    var line = 140, current = sections[0];
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].getBoundingClientRect().top <= line) current = sections[i];
    }
    // At the very bottom, the last section is what you are reading.
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
      current = sections[sections.length - 1];
    }
    for (var id in links) links[id].removeAttribute('aria-current');
    links[current.id].setAttribute('aria-current', 'true');
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { mark(); ticking = false; });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  mark();
})();
