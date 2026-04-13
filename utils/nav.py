from config import NAV_ITEMS


def is_active_path(href: str, current_path: str) -> bool:
    if href == "/":
        return current_path == "/"
    return current_path == href


def render_nav(current_path: str) -> str:
    parts = []

    for item in NAV_ITEMS:
        if "children" not in item:
            active_class = "active" if is_active_path(item["href"], current_path) else ""
            parts.append(
                f"<a class='nav-item {active_class}' href='{item['href']}'>{item['label']}</a>"
            )
            continue

        dropdown_active = any(is_active_path(child["href"], current_path) for child in item["children"])
        active_class = "active" if dropdown_active else ""

        child_links = "".join(
            f"<a class='dropdown-link {'active' if is_active_path(child['href'], current_path) else ''}' "
            f"href='{child['href']}'>{child['label']}</a>"
            for child in item["children"]
        )

        parts.append(
            f"""
            <details class="nav-dropdown {active_class}">
                <summary class="nav-item">{item['label']}</summary>
                <div class="dropdown-menu">
                    {child_links}
                </div>
            </details>
            """
        )

    return "".join(parts)