import pandas as pd
input_csv = 'teff'
output_csv1 = f'{input_csv}_1.csv'
output_csv2 = f'{input_csv}_2.csv'

df = pd.read_csv(input_csv+'.csv')
halfway = len(df) // 2
part1 = df.iloc[:halfway]
part2 = df.iloc[halfway:]
part1.to_csv(output_csv1, index=False)
part2.to_csv(output_csv2, index=False)
print(f"Split into two equal parts and saved as {output_csv1} and {output_csv2}")
