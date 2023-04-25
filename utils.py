from sqlalchemy import create_engine, Table, Column, String, Float, MetaData

def results_to_sqlite(result):
   
    engine = create_engine('sqlite:///{}.db'.format("mapping"), echo=False)
    metaadata = MetaData(engine)

    mapping = Table('mapping', metaadata,
                    Column('X (test func)', Float, primary_key=False),
                    Column('Y (test func)', Float),
                    Column('Delta Y (test func)', Float),
                    Column('No. of ideal func', String(50))
                    )

    metaadata.create_all()

    execute_map = []
    for item in result:
        point = item["point"]
        classification = item["classification"]
        delta_y = item["delta_y"]

        name_classification = None
        if classification is not None:
            name_classification = classification.name.replace("y", "N")
        else:
            
            name_classification = "-"
            delta_y = -1

        execute_map.append(
            {"X (test func)": point["x"], "Y (test func)": point["y"], "Delta Y (test func)": delta_y,
             "No. of ideal func": name_classification})

    i = mapping.insert()
    i.execute(execute_map)