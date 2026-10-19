# Sales data dictionary

## date
The day the order was placed, in YYYY-MM-DD format. One row is one order.

## category
Product category: Electronics, Clothing or Grocery.

## region
The sales team that handled the order: North or South.

## units
Number of items in the order.

## sales
Order revenue in US dollars, after discounts and before tax.

## Big order
A "big order" is an order (row) with sales of 900 dollars or more. Big orders are reviewed by a manager.

## Average order value
Average order value (AOV) means the mean of the sales column over the selected orders.

## Price per unit
Price per unit for a group of orders is total sales divided by total units, not the mean of each row's price.
