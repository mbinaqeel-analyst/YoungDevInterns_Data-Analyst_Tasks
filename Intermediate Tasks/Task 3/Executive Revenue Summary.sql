SELECT 
ROUND(SUM(quantity*list_price*(1-discount)),2) as Total_Revenue from order_items;

SELECT 
ROUND(SUM(quantity*list_price*(1-discount)),2)/ COUNT(DISTINCT (order_id), 2) AS Average_Order_Value 
from order_items;

SELECT 
	CONCAT(discount * 100, '%') AS Discount_Percentage,
    discount AS Discount_Level,
    COUNT(*) AS Number_of_Line_Items,
    SUM(quantity) AS Total_Units_Sold,
    ROUND(SUM(quantity * list_price), 2) AS Potential_Revenue,
    ROUND(SUM(quantity * list_price * (1 - discount)), 2) AS Actual_Revenue,
    ROUND(SUM(quantity * list_price) - SUM(quantity * list_price * (1 - discount)), 2) AS Total_Discount_Amount
FROM 
    order_items
GROUP BY 
    discount
ORDER BY 
    discount DESC;

