/**
 * Expand the 'Reference' section of the navigation
 * 
 * @param {boolean} remove_icon - Whether to remove the '>' icon
 */
function expandReferenceSection(remove_icon = false) {
    // Find all labels in the navigation menu
    const navLabels = document.querySelectorAll('.md-nav__item label');

    // Search for the label with the text 'Reference'
    Array.from(navLabels).forEach(label => {
        if (label.textContent.trim() === 'Reference') {
            // Find the associated checkbox to expand the section
            const toggleInput = label.previousElementSibling;
            if (toggleInput && toggleInput.tagName === 'INPUT') {
                toggleInput.checked = true;
            }

            // Find the '>' icon and hide it
            if (remove_icon) {
                const icon = label.querySelector('.md-nav__icon');
                if (icon) {
                    icon.style.display = 'none';
                }
            }
        }
    });
}

/**
 * Hides the Table of Contents (TOC) section if it only contains one link.
 */
function hideSingleItemTOC() {
    // Find the TOC list
    const tocList = document.querySelector('.md-nav--secondary .md-nav__list');

    if (tocList) {
        // Count the number of list items (links) in the TOC
        const itemCount = tocList.querySelectorAll('li').length;

        // If there is only one item, hide the entire TOC section
        if (itemCount === 1) {
            console.log("only one")
            const tocSection = document.querySelector('.md-sidebar--secondary[data-md-component="sidebar"][data-md-type="toc"]');
            if (tocSection) {
                tocSection.style.display = 'none';
            }
        }
    }
}

function main() {
    document.addEventListener("DOMContentLoaded", function () {
        // Expand the 'Reference' section of the navigation
        expandReferenceSection();

        // Hide the Table of Contents (TOC) section if it only contains one link
        hideSingleItemTOC();
    })
}

main();