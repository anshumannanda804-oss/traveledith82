// currency.js

let selectedCurrency = "INR";

/* Change Currency */
function changeCurrency(currency) {

    selectedCurrency = currency;

    // Save user preference
    localStorage.setItem("currency", currency);

    updateCurrency();
}


/* Get Converted Price */
async function convertPrice(amount) {

    try {

        const response = await fetch(`/currency/${selectedCurrency}?amount=${amount}`);

        const data = await response.json();

        return data;

    } catch (error) {

        console.error("Currency Error:", error);

        return null;
    }
}


/* Update All Prices */
async function updateCurrency() {

    const prices = document.querySelectorAll(".price");

    for (const item of prices) {

        const usd = parseFloat(item.dataset.usd);

        const result = await convertPrice(usd);

        if (result) {

            item.innerHTML = `${result.symbol}${result.price}`;
        }
    }
}


/* Restore Saved Currency */
window.addEventListener("load", () => {

    const saved = localStorage.getItem("currency");

    if (saved) {

        selectedCurrency = saved;

        const select = document.getElementById("currencySelect");

        if (select) {

            select.value = saved;
        }
    }

    updateCurrency();

});