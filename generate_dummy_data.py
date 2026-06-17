"""Generate a small Java project and expert-reference CSV for smoke tests."""

from __future__ import annotations

import csv
import os
import shutil


def create_example_project(root: str = "example_project", gt_path: str = "ground_truth.csv") -> None:
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    classes = {
        "User.java": """public class User {
    private Profile profile;
    public User(Profile profile) { this.profile = profile; }
    public Profile getProfile() { return profile; }
    public String id() { return profile.getName(); }
    public void activate() { profile.activate(); }
    public boolean isActive() { return profile != null; }
    public void update(Profile next) { this.profile = next; }
    public String describe() { return \"user\" + id(); }
    public int score() { return id().length(); }
}
""",
        "Profile.java": """public class Profile {
    private String name;
    public Profile(String name) { this.name = name; }
    public String getName() { return name; }
    public void activate() { this.name = name.trim(); }
    public boolean hasName() { return name != null; }
    public void rename(String next) { this.name = next; }
    public String label() { return \"profile\"; }
    public int size() { return name.length(); }
    public String toString() { return name; }
}
""",
        "Order.java": """public class Order {
    private User owner;
    private Product item;
    public Order(User owner, Product item) { this.owner = owner; this.item = item; }
    public User getOwner() { return owner; }
    public Product getItem() { return item; }
    public String summary() { return owner.id() + item.getId(); }
    public boolean valid() { return owner != null && item != null; }
    public void replace(Product next) { this.item = next; }
    public int quantity() { return 1; }
    public String status() { return \"created\"; }
}
""",
        "Product.java": """public class Product {
    private String id;
    public Product(String id) { this.id = id; }
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public boolean valid() { return id != null; }
    public String label() { return \"product\"; }
    public int length() { return id.length(); }
    public String toString() { return id; }
    public boolean archived() { return false; }
}
""",
        "Payment.java": """public class Payment {
    private Order order;
    public Payment(Order order) { this.order = order; }
    public Order getOrder() { return order; }
    public boolean charge() { return order.valid(); }
    public String receipt() { return order.summary(); }
    public void refund() { this.order = order; }
    public String gateway() { return \"dummy\"; }
    public int retries() { return 0; }
    public boolean settled() { return true; }
    public String status() { return \"paid\"; }
}
""",
    }
    for name, content in classes.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
            handle.write(content)
    with open(gt_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "True_Module"])
        writer.writerows([
            ["User", 0], ["Profile", 0], ["Order", 1], ["Payment", 1], ["Product", 2]
        ])
    print(f"Example project created in ./{root}")
    print(f"Ground truth written to ./{gt_path}")


if __name__ == "__main__":
    create_example_project()
