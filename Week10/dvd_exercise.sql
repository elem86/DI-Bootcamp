-- 1.
SELECT *
FROM customer;

-- 2.
SELECT CONCAT(first_name, ' ', last_name) AS full_name
FROM customer;

-- 3.
SELECT DISTINCT create_date
FROM customer;

-- 4.
SELECT *
FROM customer
ORDER BY first_name DESC;

-- 5.
SELECT film_id, title, description, release_year, rental_rate
FROM film
ORDER BY rental_rate ASC;

-- 6.
-- There is no phone number or district in the database. 

-- 7.
SELECT *
FROM film
WHERE film_id = 15 OR film_id = 150;

-- 8.
SELECT film_id, title, description, length, rental_rate
FROM film
WHERE title = 'Rain Man';

-- 9.
SELECT film_id, title, description, length, rental_rate
FROM film
WHERE title ILIKE 'Ra%';

-- 10.
SELECT title, rental_rate
FROM film
ORDER BY rental_rate ASC
LIMIT 10;

-- 11.
SELECT film_id, title, rental_rate
FROM film
ORDER BY rental_rate ASC
OFFSET 10 ROWS
FETCH NEXT 10 ROWS ONLY;

-- 12.
SELECT
    customer.customer_id,
    customer.first_name,
    customer.last_name,
    payment.amount,
    payment.payment_date
FROM customer
JOIN payment
    ON customer.customer_id = payment.customer_id
ORDER BY customer.customer_id;

-- 13.
SELECT
    film.film_id,
    film.title
FROM film
LEFT JOIN inventory
    ON film.film_id = inventory.film_id
WHERE inventory.inventory_id IS NULL;

-- 14.
SELECT
    city.city,
    country.country
FROM city
JOIN country
    ON city.country_id = country.country_id
ORDER BY country.country, city.city;


-- 15.
SELECT
    customer.customer_id,
    customer.first_name,
    customer.last_name,
    payment.amount,
    payment.payment_date,
    payment.staff_id
FROM customer
JOIN payment
    ON customer.customer_id = payment.customer_id
ORDER BY payment.staff_id, customer.customer_id;


