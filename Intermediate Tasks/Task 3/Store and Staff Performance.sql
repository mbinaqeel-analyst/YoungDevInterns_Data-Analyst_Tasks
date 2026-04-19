SELECT 
    s.store_name,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Store_Revenue
FROM 
    stores s
INNER JOIN orders o ON s.store_id = o.store_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 
    s.store_name
ORDER BY 
    Store_Revenue DESC;
    
SELECT 
    CONCAT(st.first_name, ' ', st.last_name) AS Staff_Member,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Sales_Value
FROM 
    staffs st
INNER JOIN orders o ON st.staff_id = o.staff_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 
    st.staff_id, st.first_name, st.last_name
ORDER BY 
    Total_Sales_Value DESC;
    
SELECT 
    CONCAT(m.first_name, ' ', m.last_name) AS Manager_Name,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Team_Revenue
FROM 
    staffs s
INNER JOIN staffs m ON s.manager_id = m.staff_id
INNER JOIN orders o ON s.staff_id = o.staff_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 
    m.staff_id, m.first_name, m.last_name
ORDER BY 
    Team_Revenue DESC;