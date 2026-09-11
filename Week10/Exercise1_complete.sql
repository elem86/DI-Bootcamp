-- Exercise 1: Items and Customers

-- Create tables
CREATE TABLE items (
    item_id INTEGER PRIMARY KEY,
    item_name VARCHAR(100) NOT NULL,
    price INTEGER NOT NULL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL
);

-- Insert items
INSERT INTO items (item_id, item_name, price)
VALUES
    (1, 'Small Desk', 100),
    (2, 'Large Desk', 300),
    (3, 'Fan', 80);

-- Insert customers
INSERT INTO customers (customer_id, first_name, last_name)
VALUES
    (1, 'Greg', 'Jones'),
    (2, 'Sandra', 'Jones'),
    (3, 'Scott', 'Scott'),
    (4, 'Trevor', 'Green'),
    (5, 'Melanie', 'Johnson');

-- Exercise 1.1
SELECT *
FROM customers, items;

-- Exercise 1.2
SELECT *
FROM items
WHERE price > 80;

-- Exercise 1.3
SELECT *
FROM items
WHERE price <= 300;

-- Exercise 1.4
SELECT *
FROM customers
WHERE last_name = 'Smith';

-- Exercise 1.5
SELECT *
FROM customers
WHERE last_name = 'Jones';

-- Exercise 1.6
SELECT *
FROM customers
WHERE first_name <> 'Scott';
