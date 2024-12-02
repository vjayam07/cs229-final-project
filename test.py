import osmnx as ox

countries_with_street_view_coverage = [
    # List of countries (same as above)
    "Botswana", "Ghana", "Kenya", "Lesotho", "Madagascar", "Nigeria", "Rwanda", "Senegal", "South Africa", "Uganda",
    "Bangladesh", "Hong Kong", "India", "Indonesia", "Israel", "Japan", "Jordan", "Laos", "Macau", "Malaysia",
    "Mongolia", "Nepal", "Philippines", "Singapore", "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Vietnam",
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany", "Greece", "Hungary",
    "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia",
    "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom",
    "Vatican City", "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda", "Canada",
    "Cayman Islands", "Costa Rica", "Curaçao", "Dominica", "Dominican Republic", "El Salvador", "Greenland", "Grenada",
    "Guadeloupe", "Guatemala", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Nicaragua", "Panama",
    "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Sint Maarten",
    "Trinidad and Tobago", "United States", "U.S. Virgin Islands", "Australia", "Fiji", "French Polynesia", "Guam",
    "New Caledonia", "New Zealand", "Northern Mariana Islands", "Papua New Guinea", "Samoa", "Argentina", "Bolivia",
    "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay"
]

unrecognized_countries = []

for country in countries_with_street_view_coverage:
    try:
        # Test if the country name is recognized by OSMnx
        ox.geocoder.geocode_to_gdf(country)
        print(f"Recognized: {country}")
    except Exception:
        print(f"Unrecognized: {country}")
        unrecognized_countries.append(country)

print("\nUnrecognized Countries:", unrecognized_countries)