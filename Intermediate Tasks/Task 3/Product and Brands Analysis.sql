SELECT 
    p.product_name,
    SUM(oi.quantity) AS Units_Sold,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Revenue
FROM 
    products p
INNER JOIN 
    order_items oi ON p.product_id = oi.product_id
GROUP BY 
    p.product_name
ORDER BY 
    Units_Sold DESC
LIMIT 10;

SELECT 
    c.category_name,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Revenue
FROM 
    categories c
INNER JOIN 
    products p ON c.category_id = p.category_id
INNER JOIN 
    order_items oi ON p.product_id = oi.product_id
GROUP BY 
    c.category_name
ORDER BY 
    Total_Revenue DESC;
    
SELECT 
    b.brand_name,
    SUM(oi.quantity) AS Total_Units_Sold,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Revenue
FROM 
    brands b
INNER JOIN 
    products p ON b.brand_id = p.brand_id
INNER JOIN 
    order_items oi ON p.product_id = oi.product_id
GROUP BY 
    b.brand_name
ORDER BY 
    Total_Revenue DESC;
    
SELECT 
    p.model_year,
    SUM(oi.quantity) AS Units_Sold,
    ROUND(SUM(oi.quantity * oi.list_price * (1 - oi.discount)), 2) AS Total_Revenue
FROM 
    products p
INNER JOIN 
    order_items oi ON p.product_id = oi.product_id
GROUP BY 
    p.model_year
ORDER BY 
    p.model_year DESC;