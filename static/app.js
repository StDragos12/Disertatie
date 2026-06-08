document.addEventListener("DOMContentLoaded", () => {
  const revealElements = document.querySelectorAll(".reveal");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("active");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
      rootMargin: "0px 0px -40px 0px",
    }
  );

  revealElements.forEach((el) => {
    if (!el.classList.contains("active")) {
      observer.observe(el);
    }
  });

  const magnetics = document.querySelectorAll(".magnetic");

  magnetics.forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;

      el.style.transform = `translate(${x * 0.04}px, ${y * 0.04}px)`;
    });

    el.addEventListener("mouseleave", () => {
      el.style.transform = "";
    });
  });
});
document.addEventListener("submit", function (event) {
    const form = event.target;

    if (!form.classList.contains("dataset-upload-form")) {
        return;
    }

    const button = form.querySelector("button[type='submit']");

    if (button) {
        button.disabled = true;
        button.textContent = "Se încarcă...";
        button.classList.add("is-loading");
    }
});