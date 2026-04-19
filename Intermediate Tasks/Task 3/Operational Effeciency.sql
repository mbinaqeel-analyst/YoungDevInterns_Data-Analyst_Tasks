SELECT 
    ROUND(AVG(DATEDIFF(shipped_date, order_date)), 1) AS Avg_Days_To_Ship
FROM 
    orders
WHERE 
    shipped_date IS NOT NULL; 
    
SELECT 
    order_id, 
    customer_id, 
    order_date, 
    required_date, 
    shipped_date,
    DATEDIFF(shipped_date, required_date) AS Days_Late
FROM 
    orders
WHERE 
    shipped_date > required_date
ORDER BY 
    Days_Late DESC;