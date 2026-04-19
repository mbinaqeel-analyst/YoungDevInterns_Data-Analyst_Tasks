SELECT 
    CONCAT(c.first_name, ' ', c.last_name) AS Customer_Name,
    COUNT(DISTINCT o.order_id) AS Total_Orders,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Spent
FROM 
    customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 
    c.customer_id, c.first_name, c.last_name
ORDER BY 
    Total_Spent DESC
LIMIT 10;

SELECT 
    c.state, 
    c.city,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Revenue
FROM 
    customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 
    c.state, c.city
ORDER BY 
    Revenue DESC;