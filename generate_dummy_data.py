import os

def create_example_project():
    os.makedirs("example_project", exist_ok=True)
    classes = {
        "User.java": "public class User { private Profile profile; }",
        "Profile.java": "public class Profile { private String name; }",
        "Order.java": "public class Order { private User owner; private Product item; }",
        "Product.java": "public class Product { private String id; }",
        "Payment.java": "public class Payment { private Order order; }"
    }
    for name, content in classes.items():
        with open(f"example_project/{name}", "w") as f:
            f.write(content)
    print("Example project created in './example_project'")

if __name__ == "__main__":
    create_example_project()