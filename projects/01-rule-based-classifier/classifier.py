def classify_customer(total_spent, orders, days_since_last_purchase):
        if not isinstance(total_spent, (int,float)) or not isinstance(orders, int) or not isinstance(days_since_last_purchase, int):
             raise TypeError("TypeError")
        elif isinstance(total_spent, bool) or isinstance(orders, bool) or isinstance(days_since_last_purchase, bool):
            raise TypeError("Invalid Input")
        if total_spent >= 0 and orders >= 0 and days_since_last_purchase >= 0:
                if total_spent > 1000 and orders > 10 and days_since_last_purchase <= 30:
                    return "VIP"
                elif days_since_last_purchase > 90:
                    return "At Risk"
                else: 
                    return "Regular"
        else:
            raise ValueError("ValueError")
