const days_per_year = 365.2422
const global_base_date = Date(2000,1,1)
const global_base_date_as_day = convert(Dates.Day, global_base_date)

"""
    years_between(a::Date, b::Date)
    years_between(a::Dates.Day, b::Dates.Day)

Returns the number of years between two dates. For the purposes of this calculation
there are 365.2422 days in a year.
"""
function years_between(a::Date, b::Date)
    return (Dates.days(a) -Dates.days(b))/ days_per_year
end
function years_between(a::Dates.Day, b::Dates.Day)
    return (convert(Int, a)-convert(Int, b))/ days_per_year
end

"""
    years_between(a::Date, b::Date)
    years_between(a::Dates.Day, b::Dates.Day)

Returns the number of years that have elapsed since 1-Jan-2000. For the purposes of this calculation
there are 365.2422 days in a year.
"""
function years_from_global_base(a::Date)
    return years_between(a, global_base_date)
end

function years_from_global_base(a::Dates.Day)
    return years_between(a, global_base_date_as_day)
end
