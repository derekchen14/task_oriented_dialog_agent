I. Restaurant Domain:
  - reservation subdomain
    inform dialogue act
      inform(area=north), inform(area=east), inform(area=south)
      inform(price=cheap), inform(price=moderate), inform(price=expensive)
      inform(food=chinese), inform(food=italian), inform(food=japanese)
    question dialogue act
      question(address=the_missing_sock), question(address=il_casaro)
      question(phone=the_missing_sock), question(phone=il_casaro)
      question(rating=the_missing_sock), question(rating=il_casaro)
    request dialogue act
      request(reserve=the_missing_sock), request(reserve=il_casaro)
      request(call=the_missing_sock), request(call=il_casaro)
  - menu subdomain
    inform dialogue act
      inform(dish=kung_pao_chicken), inform(dish=pepperoni_pizza)
      inform(drink=lemonade), inform(drink=diet_soda), inform(drink=water)
      inform(appetizer=takoyaki), inform(appetizer=green_gem_salad)
    question dialogue act
      question(vegetarian=the_missing_sock), question(vegetarian=il_casaro)
II. Transportation Domain:
  - traffic subdomain
    question dialogue act
      question(location=101_freeway), question(location=san_mateo)
      question(time=now), question(time=tomorrow), question(time=1700)
    inform dialogue act
      inform(time=night), inform(start_loc=current), inform(end_loc=san_jose)
  - weather subdomain
    question dialgoue act
      question(location=mountain_view), question(location=san_mateo)
      question(time=now), question(time=tomorrow), question(time=1700)
III. Hotel Domain:
  - reservation subdomain
  - front desk subdomain
IV. Calendar Domain:
  - meetings subdomain (for business)
  - event subdomain (for pleasure)
V. Airport Domain:
  - reservation subdomain
  - airline subdomain
VI. Internet Service Provider Domain:
  - sales subdomain
  - customer support subdomain
  - troubleshoot subdomain