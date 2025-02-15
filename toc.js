// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="index.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">User Guide</li><li class="chapter-item expanded "><a href="guide/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="guide/hello_world.html"><strong aria-hidden="true">2.</strong> Hello World - MNIST</a></li><li class="chapter-item expanded "><a href="guide/cheatsheet.html"><strong aria-hidden="true">3.</strong> PyTorch cheatsheet</a></li><li class="chapter-item expanded affix "><li class="part-title">Reference Guide</li><li class="chapter-item expanded "><a href="inference/inference.html"><strong aria-hidden="true">4.</strong> Running a model</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="inference/hub.html"><strong aria-hidden="true">4.1.</strong> Using the hub</a></li></ol></li><li class="chapter-item expanded "><a href="error_manage.html"><strong aria-hidden="true">5.</strong> Error management</a></li><li class="chapter-item expanded "><a href="training/training.html"><strong aria-hidden="true">6.</strong> Training</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="training/simplified.html"><strong aria-hidden="true">6.1.</strong> Simplified</a></li><li class="chapter-item expanded "><a href="training/mnist.html"><strong aria-hidden="true">6.2.</strong> MNIST</a></li><li class="chapter-item expanded "><div><strong aria-hidden="true">6.3.</strong> Fine-tuning</div></li><li class="chapter-item expanded "><div><strong aria-hidden="true">6.4.</strong> Serialization</div></li></ol></li><li class="chapter-item expanded "><div><strong aria-hidden="true">7.</strong> Advanced Cuda usage</div></li><li><ol class="section"><li class="chapter-item expanded "><div><strong aria-hidden="true">7.1.</strong> Writing a custom kernel</div></li><li class="chapter-item expanded "><div><strong aria-hidden="true">7.2.</strong> Porting a custom kernel</div></li></ol></li><li class="chapter-item expanded "><div><strong aria-hidden="true">8.</strong> Using MKL</div></li><li class="chapter-item expanded "><div><strong aria-hidden="true">9.</strong> Creating apps</div></li><li><ol class="section"><li class="chapter-item expanded "><div><strong aria-hidden="true">9.1.</strong> Creating a WASM app</div></li><li class="chapter-item expanded "><div><strong aria-hidden="true">9.2.</strong> Creating a REST api webserver</div></li><li class="chapter-item expanded "><div><strong aria-hidden="true">9.3.</strong> Creating a desktop Tauri app</div></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
