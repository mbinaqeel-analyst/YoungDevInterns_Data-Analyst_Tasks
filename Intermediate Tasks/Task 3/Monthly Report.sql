SELECT 
    DATE_FORMAT(o.order_date, '%Y-%m') AS Order_Month,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Revenue
FROM 
    orders o
INNER JOIN 
    order_items oi ON o.order_id = oi.order_id
GROUP BY 
    Order_Month
ORDER BY 
    Order_Month ASC;