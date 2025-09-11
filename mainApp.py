print("\n#################################################")
print("     Welcome to HC Order status application")
print("#################################################\n")

def get_order_status(order_id):
    # Simulate fetching order status from a database or API
    order_statuses = {
        1: "Processing",
        2: "Shipped",
        3: "Delivered",
        4: "Cancelled"
    }
    return order_statuses.get(order_id, "Order ID not found")

if __name__ == "__main__":
    order_id = int(input("Enter your order ID: "))
    status = get_order_status(order_id)
    print(f"Order ID {order_id} status: {status}")
