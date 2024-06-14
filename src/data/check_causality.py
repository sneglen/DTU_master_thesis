from src.spe.agents.pydantic_annotator_classes import fun_choices_object as fun_class 


def check_causality(articles, logger):

  # list with all checks instead of "if True"
  check_1a = 0 # Any PER entry in ORG list?
  check_1b = 0 # Any ORG entry in PER list?
  check_2  = 0 # Is SRC always in {PER, ORG, empty}?
  check_3a = 0 # GEN: always M/F if SRC = single PER?
  check_3b = 0 # GEN: always  X  if SRC = single ORG?
  check_3c = 0 # GEN: always  X  if SRC = empty?
  check_4a = 0 # FUN: Distribution of the 8 options when SRC conditioned to be contained in PER
  check_4b = 0 # FUN: Distribution of the 8 options when SRC conditioned to be contained in ORG
  check_4c = 0 # FUN: Distribution of the 8 options when SRC conditioned to be emtpy
  check_5  = 0 # ROL: TBD
  check_6  = 0 # EMP: Distribution between PER, ORG, empty, non-empty string?


## Any PER entry in ORG list?
  if check_1a: 
    logger.notice("\n## 1a: Any PER entry in ORG list?")
    entry_found = False
    for idx, article in enumerate(articles):
      if len(article['PER']) > 0 and article['PER'] != ['']:
        for per in article['PER']:
            logger.notice(f"    Article[{idx}]: PER '{per}' is also in ORG")
            logger.notice(f"PER: {article['PER']}") 
            logger.notice(f"ORG: {article['ORG']}")
            entry_found = True

    if not entry_found:
      logger.notice("   No PER entry in ORG list found")

## Any ORG entry in PER list?
  if check_1b: 
    logger.notice("\n## 1b: Any ORG entry in PER list?")
    entry_found = False
    for idx, article in enumerate(articles):
      if len(article['ORG']) > 0  and article['ORG'] != ['']:
        for org in article['ORG']:
          if org in article['PER']:
            logger.notice(f"    Article[{idx}]: ORG '{org}' is also in PER")
            logger.notice(f"PER: {article['PER']}") 
            logger.notice(f"ORG: {article['ORG']}")
            entry_found = True

    if not entry_found:
      logger.notice("   No ORG entry in PER list found")


## Is SRC always in {PER, ORG, empty}?
  if check_2:
    per_count = 0
    org_count = 0
    empty_count = 0
    logger.notice("\n## 2: Is SRC always in {PER, ORG, empty}?")
    entry_found = False
    for idx, article in enumerate(articles):
      per_list = article['PER']
      org_list = article['ORG']
      if len(article['QUOTES']) > 0:
        for quote_idx, quote in enumerate(article['QUOTES']):
          for src in quote['SRC']:
            if src in per_list:
              per_count += 1
            elif src in org_list:
              org_count += 1
            elif src == "" or src == []:
              empty_count += 1
            if src not in per_list and src not in org_list and src != "":
              logger.notice(f"    Article[{idx}]Quote[{quote_idx}]: SRC '{src}' is not in {['PER', 'ORG', '']}")
              entry_found = True

    if not entry_found:
      logger.notice("   SRC is always in {PER, ORG, empty}")

    logger.notice(f"PER:   {per_count:<4} {per_count/(per_count+org_count+empty_count)*100:.1f}%")
    logger.notice(f"ORG:   {org_count:<4} {org_count/(per_count+org_count+empty_count)*100:.1f}%")
    logger.notice(f"empty: {empty_count:<4} {empty_count/(per_count+org_count+empty_count)*100:.1f}%")

## GEN: always M/F if SRC = single PER?
  if check_3a:
    logger.notice("\n## 3a: GEN: always M/F if SRC = single PER?")
    entry_found = False
    for idx, article in enumerate(articles):
      per_list = article['PER']
      org_list = article['ORG']
      if len(article['QUOTES']) > 0:
        for quote_idx, quote in enumerate(article['QUOTES']):
          if len(quote['SRC']) == 1:
            src = quote['SRC'][0]
            if src in per_list:
              if quote['GEN'] not in ['M', 'F']:
                logger.notice(f"    Article[{idx}]Quote[{quote_idx}]: SRC '{src}' is in PER but GEN is not M/F")
                logger.notice(f"GEN: {quote['GEN']}")
                entry_found = True
    
    if not entry_found:
      logger.notice("   GEN is always M/F if SRC = single PER")

## GEN: always  X  if SRC = single ORG?
  if check_3b:
    logger.notice("\n## 3b: GEN: always  X  if SRC = single ORG?")
    entry_found = False
    for idx, article in enumerate(articles):
      per_list = article['PER']
      org_list = article['ORG']
      if len(article['QUOTES']) > 0:
        for quote_idx, quote in enumerate(article['QUOTES']):
          if len(quote['SRC']) == 1:
            src = quote['SRC'][0]
            if src in org_list:
              if quote['GEN'] != 'X':
                logger.notice(f"    Article[{idx}]Quote[{quote_idx}]: SRC '{src}' is in ORG but GEN is not X")
                logger.notice(f"GEN: {quote['GEN']}")
                entry_found = True
    
    if not entry_found:
      logger.notice("   GEN is always X if SRC = single ORG")

## GEN: always  X  if SRC = empty?
  if check_3c:
    logger.notice("\n## 3c: GEN: always  X  if SRC = empty?")
    entry_found = False
    for idx, article in enumerate(articles):
      per_list = article['PER']
      org_list = article['ORG']
      if len(article['QUOTES']) > 0:
        for quote_idx, quote in enumerate(article['QUOTES']):
          if len(quote['SRC']) == 0:
            if quote['GEN'] != 'X':
              logger.notice(f"    Article[{idx}]Quote[{quote_idx}]: SRC is empty but GEN is not X")
              logger.notice(f"GEN: {quote['GEN']}")
              entry_found = True
    
    if not entry_found:
      logger.notice("   GEN is always X if SRC = empty")

## FUN: Distribution of the 8 options when SRC conditioned to be contained in PER
  if check_4a:
    logger.notice("\n## 4a: FUN_dist[SRC|PER]")

    fun_distribution = {choice: 0 for choice in fun_class}
    total_count = 0

    for idx, article in enumerate(articles):
        per_list = article['PER']
        if 'QUOTES' in article and len(article['QUOTES']) > 0:
            for quote_idx, quote in enumerate(article['QUOTES']):
              if len(quote['SRC']) >= 1:
                src = quote['SRC'][0]
                if src in per_list: 
                  fun = quote['FUN']
                  if fun in fun_distribution:
                      fun_distribution[fun] += 1
                      total_count += 1

    # Printing the distribution
    for fun, count in fun_distribution.items():
      percentage = (count / total_count) * 100 if total_count > 0 else 0
      print(f"{fun:<22}: {count:<3} ({percentage:.1f}%)")

## FUN: Distribution of the 8 options when SRC conditioned to be contained in ORG
  if check_4b:
    logger.notice("\n## 4b: FUN_dist[SRC|ORG]")

    fun_distribution = {choice: 0 for choice in fun_class}
    total_count = 0

    for idx, article in enumerate(articles):
        org_list = article['ORG']
        if 'QUOTES' in article and len(article['QUOTES']) > 0:
            for quote_idx, quote in enumerate(article['QUOTES']):
              if len(quote['SRC']) >= 1:
                src = quote['SRC'][0]
                if src in org_list: 
                  fun = quote['FUN']
                  if fun in fun_distribution:
                      fun_distribution[fun] += 1
                      total_count += 1

    # Printing the distribution
    for fun, count in fun_distribution.items():
      percentage = (count / total_count) * 100 if total_count > 0 else 0
      print(f"{fun:<22}: {count:<3} ({percentage:.1f}%)")

## FUN: Distribution of the 8 options when SRC conditioned to be empty
  if check_4c:
    logger.notice("\n## 4c: FUN_dist[SRC|empty]")

    fun_distribution = {choice: 0 for choice in fun_class}
    total_count = 0

    for idx, article in enumerate(articles):
        if 'QUOTES' in article and len(article['QUOTES']) > 0:
            for quote_idx, quote in enumerate(article['QUOTES']):
              if len(quote['SRC']) == 0:
                fun = quote['FUN']
                if fun in fun_distribution:
                    fun_distribution[fun] += 1
                    total_count += 1

    # Printing the distribution
    for fun, count in fun_distribution.items():
      percentage = (count / total_count) * 100 if total_count > 0 else 0
      print(f"{fun:<22}: {count:<3} ({percentage:.1f}%)")

## EMP: Distribution between PER, ORG, empty, non-empty string?
  if check_6: 
    logger.notice("\n## 6: EMP distribution between PER, ORG, empty, non-empty string")
    per_count = 0
    org_count = 0
    empty_count = 0
    non_empty_count = 0
    entry_found = False
    for idx, article in enumerate(articles):
      per_list = article['PER']
      org_list = article['ORG']
      if 'QUOTES' in article and len(article['QUOTES']) > 0:
        for quote_idx, quote in enumerate(article['QUOTES']):
            if quote['EMP'] == []:
                empty_count += 1
            else:
              for emp in quote['EMP']:
                if emp in per_list:
                  per_count += 1
                elif emp in org_list:
                  org_count += 1
                else:
                  non_empty_count += 1  # Assumes any other non-empty value falls here

    logger.notice(f"PER:   {per_count:<4} {per_count/(per_count+org_count+empty_count+non_empty_count)*100:.1f}%")
    logger.notice(f"ORG:   {org_count:<4} {org_count/(per_count+org_count+empty_count+non_empty_count)*100:.1f}%")
    logger.notice(f"empty: {empty_count:<4} {empty_count/(per_count+org_count+empty_count+non_empty_count)*100:.1f}%")
    logger.notice(f"non-empty: {non_empty_count:<4} {non_empty_count/(per_count+org_count+empty_count+non_empty_count)*100:.1f}%")
