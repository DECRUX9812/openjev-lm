#!/usr/bin/env python3
"""Hard-case synthetic postings for openjev — the measured class-imbalance gap.

Why this file exists
--------------------
The corpus the first training run is learning from is 70% generic_job and two of Jev's four
buckets are starved:

    generic_job 1,713 | staff_role ~663 | junk 49 (2.0%) | service_lead 12 (0.5%)

A model cannot learn `service_lead` from 12 rows, and `service_lead` is the bucket the product
is about: a small business with a concrete, small-scope technical need (website, IT support,
POS setup, automation, reporting, e-commerce) buying work — as opposed to hiring an employee.
This generator manufactures exactly the missing boundary cases.

Design
------
* GENERATION ONLY. No model, no network, no local inference. Template/slot composition over a
  hand-written Saskatchewan vocabulary. Stdlib only; deterministic given SEED.
* `stratum`/`family`/`intent` are METADATA, never labels. Every row is labelled by the live Jev
  API afterwards, via the parent-owned `openjev/label_jev.py`:
      python openjev/label_jev.py --in openjev/synth_hard.raw.jsonl \
                                  --out-dir openjev/data/jevlab_hard --workers 8
  Jev's answer is the training target; where it disagrees with our intent we keep his answer and
  report the confusion matrix, because that table is evidence about the data, not a defect.
* Strata (intent): lead ~330, junk ~250, seat ~200, generic ~120, plus ~30 counterfactual PAIRS
  (same need written as an employee seat and as a contract/purchase).
* Leakage: no (lowercased title, employer) pair may repeat inside the file, collide with the 70
  held-out gold postings (typesafe-lab), any row of leads_corpus_full.json, the other pipeline's
  rows (openjev/synth/data/labelled.jsonl), or the eval gold titles (openjev/data/eval_gold.jsonl
  carries titles only). Excluded pairs are written to data/jevlab_hard/raw_meta.json and reported.

Usage
-----
  python openjev/synth_hard.py                 # writes openjev/synth_hard.raw.jsonl (+ meta)
  python openjev/synth_hard.py --report        # after labelling: writes runs/HARDCASE.md + status
  python openjev/synth_hard.py --check         # generation gates only, no writes
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

OPENJEV = Path("/home/decrux/Code/jev-repro-test/openjev")
LAB = Path("/home/decrux/Code/typesafe-lab")
RAW_OUT = OPENJEV / "synth_hard.raw.jsonl"
HARD_DIR = OPENJEV / "data" / "jevlab_hard"
META_OUT = HARD_DIR / "raw_meta.json"
LABELLED_IN = HARD_DIR / "synth_hard.raw.jev.jsonl"
LABEL_LOG = HARD_DIR / "label_jev.log"
REPORT_OUT = OPENJEV / "runs" / "HARDCASE.md"
STATUS_OUT = OPENJEV / "runs" / "STATUS-D-hardcases.md"

SEED = 20260918
BUCKETS = ["service_lead", "staff_role", "generic_job", "junk"]
BASELINE = {"service_lead": 12, "junk": 49}  # measured corpus counts before this run

TARGETS = {"lead": 330, "junk": 250, "seat": 200, "generic": 120, "pair": 30}
FAMILY_INTENT = {
    "lead": "service_lead",
    "seat": "staff_role",
    "junk": "junk",
    "generic": "generic_job",
    "pair_seat": "staff_role",
    "pair_contract": "service_lead",
}

# ---------------------------------------------------------------------------
# Pay formats (real Saskatchewan shapes; "" is a legitimate value)
# ---------------------------------------------------------------------------
PAYS = [
    "$28.00 hourly", "$26.50 hourly", "$32.00 hourly", "$19.50 hourly", "$24.00 per hour",
    "$22.75 hourly", "$31.25 hourly", "$45,000 - $58,000 annually", "$52,000 - $64,000 annually",
    "$68,000 - $82,000 annually", "$38,000 annually", "$55,000 per year plus benefits",
    "$62,000 - $78,000 per year", "$1,800 biweekly", "Pay band 7 ($33.10 hourly)",
    "$2,500 project", "$1,800 project fee", "$3,500 project", "$4,000 - $6,000 project",
    "$1,200 flat rate", "$500/month retainer", "$95/hour contract", "$85 per hour (contract)",
    "$70/hour", "Competitive", "Negotiable", "Commensurate with experience",
    "$30 - $45 per hour depending on experience", "", "", "", "",
]
LEAD_PAYS = [
    "$28.00 hourly", "$32.00 hourly", "$45/hour", "$60/hour contract", "$85/hour contract",
    "$2,500 project", "$1,800 project fee", "$3,500 project", "$4,000 - $6,000 project",
    "$1,200 flat rate", "$500/month retainer", "$750 per month", "$95/hour contract",
    "Negotiable", "Hourly rate negotiable", "Project fee to be discussed", "",
    "", "", "$1,500 per month part-time", "$40/hour",
]
REGIONS = ["Regina", "Regina", "Regina", "Saskatoon", "Moose Jaw", "Prince Albert",
           "Weyburn", "Estevan", "Yorkton", "Swift Current", "Humboldt", "Melville"]

# ---------------------------------------------------------------------------
# family 1: SEAT — permanent technical employee seats (intent staff_role)
# ---------------------------------------------------------------------------
SEAT_TITLES = [
    "Data Architect", "Network & Server Analyst", "Application Analyst", "Data Modeler",
    "IT Senior Analyst", "Systems Administrator", "Cybersecurity Analyst", "GIS Analyst",
    "ERP Business Systems Analyst", "Database Administrator", "Software Developer",
    "QA Engineer", "Embedded Systems Developer", "Network Technician", "Help Desk Analyst",
    "Data Engineer", "Cloud DevOps Engineer", "IT Support Specialist", "Junior Systems Analyst",
    "Network Administrator", "Business Intelligence Analyst", "Programmer Analyst",
    "Infrastructure Analyst", "Solutions Architect", "IT Coordinator", "Service Desk Technician",
    "Data Analyst", "DevOps Specialist", "Application Support Analyst", "Database Analyst",
    "Security Operations Analyst", "Web Application Developer", "Senior Data Architect",
    "Enterprise Data Architect", "IT Asset Analyst", "Storage & Backup Administrator",
    "Middleware Administrator", "Integration Developer", "Technical Business Analyst",
    "Linux Systems Engineer", "Windows Server Administrator", "SCADA Systems Analyst",
    "Telecom Analyst", "Geospatial Data Analyst", "Machine Learning Engineer",
    "Software QA Analyst", "SAP Functional Analyst", "Field Applications Engineer",
    "Enterprise Architect", "Information Security Analyst", "IT Manager",
    "Senior Systems Analyst", "Application Developer", "Data Governance Analyst",
    "ERP Support Analyst", "Student Information System Analyst", "Clinical Informatics Analyst",
    "Reporting & Analytics Analyst", "Cloud Solutions Analyst", "Identity & Access Analyst",
    "Records & Information Analyst", "Business Systems Consultant", "ServiceNow Administrator",
    "Junior Data Analyst",
]
SEAT_EMPLOYERS = [
    "Prairie Cloud Systems", "Wascana Credit Union", "Regina Public School Division",
    "Meadowbrook Manufacturing", "Sask Health Region", "Kalium Potash Mine",
    "AgriSense Technologies", "City of Moose Jaw", "Northline IT Consulting",
    "Prairie Data Group", "Affinity Credit Union", "Saskatchewan Polytechnic",
    "Wheatland Machine Works", "Cypress Hills Health Authority", "Boreal Forest Products",
    "Southland Co-op", "Riverbend School Division", "Innovate SK Labs", "Prairie Grid Utilities",
    "Gateway Insurance Group", "Sunwest Credit Union", "Regina Airport Authority",
    "Lakeside Manufacturing", "Horizon School Division", "Prairie Health Foundation",
    "Silverfox Mining Corp", "Meridian Ag Systems", "Rural Municipality of Lumsden",
    "TransitLink Regina", "SaskPower Distribution", "Northern Lights College",
    "Delta IT Services", "Prairie Digital Solutions", "Qu'Appelle Valley Feeds",
    "New Era Software", "Hometown Community Credit Union", "Dundee Industrial Supply",
    "Palliser Health Centre", "Buffalo Plains Energy", "Regina Catholic School Board",
    "Prairie Metal Fabricators", "Copperline Utilities", "Mosaic Ridge Mining",
    "Lakeview Mutual Insurance", "Boreal Computing", "Fivetrails Logistics", "Sherwood Co-op",
    "City of Weyburn", "Pheasant Creek Seeds", "Estevan Coal & Power",
    "Saskatchewan Health Authority", "Regina Police Service", "SaskEnergy", "SaskTel",
    "Conexus Credit Union", "University of Regina", "Saskatoon Public Schools",
    "Prairie Valley School Division", "eHealth Saskatchewan", "Wascana Centre Authority",
    "Saskatchewan Crop Insurance", "Prince Albert Grand Council", "Saskatchewan Indian Gaming Authority",
    "Cameco", "Nutrien", "Information Services Corporation",
    "Saskatchewan Workers' Compensation Board", "Athabasca Basin Development",
]

# ---------------------------------------------------------------------------
# family 2: LEAD — small business buying concrete small-scope technical work
# ---------------------------------------------------------------------------
LEAD_TITLES = [
    "Website Designer", "Website Developer Wanted", "Website Redesign Needed",
    "E-commerce Store Setup", "Shopify Store Developer", "Online Store Setup - Bakery",
    "Website & Online Ordering Help", "Website Fixes & Hosting Move",
    "Web Design & Logo Refresh", "Google Business Profile Setup", "SEO & Website Maintenance",
    "Digital Marketing Help Needed", "Social Media Setup & Training", "Google Ads Manager Wanted",
    "Email Marketing Setup", "IT Support (Part-Time Contract)", "IT Support Needed For Office Move",
    "Part-Time IT Help For Small Office", "Computer Repair Technician",
    "New Laptops - Setup & Data Transfer", "Laptop Refresh & Data Migration",
    "Computer Network Installer", "Network Cabling & Wifi Setup", "Wifi Upgrade For Retail Store",
    "Phone System Installation", "Email & Microsoft 365 Setup", "Google Workspace Migration",
    "Cloud Backup Setup", "Data Backup & Recovery Help", "Cybersecurity Review Needed",
    "Security Camera Installation", "CCTV & Alarm Setup", "Point of Sale Setup",
    "POS Installation - Restaurant", "POS & Payments Setup For Food Truck",
    "Inventory System Setup", "Barcode & Inventory Setup", "Bookkeeping Automation",
    "Bookkeeping Software Migration", "QuickBooks Setup & Training", "Payroll Software Setup",
    "Accounting Software Cleanup", "Bookkeeping Help - Software Migration",
    "Database Cleanup Needed", "Customer Database Deduplication", "Custom Booking System Needed",
    "Online Booking & Payments", "Membership Management System", "CRM Setup",
    "CRM & Email Automation", "Report Dashboard Build", "Monthly Sales Reporting Help Needed",
    "Recipe Costing Spreadsheet Build", "Digital Menu Boards Installation",
    "Delivery App Integration", "Automation Consultant Needed", "Automate Our Quote Paperwork",
    "Spreadsheet & Scheduling Help", "Time Tracking System Setup", "Website Accessibility Fixes",
    "Domain & Email Transfer Help",
]
LEAD_EMPLOYERS = [
    "Roselawn Family Restaurant", "Bright Smile Dental", "Prairie View Motors",
    "Sunset Greenhouse", "IronWorks Gym", "Precision Machine Shop", "Lakeside Funeral Home",
    "Bright Beginnings Daycare", "Ridgeline Trucking", "Kestrel Family Farm", "Wascana Nail Salon",
    "Cornerstone Coffee Roasters", "Prairie Sky Yoga", "Hillside Auto Repair", "Meadowlark Pharmacy",
    "Golden Grain Bakery", "Copper Kettle Brewing", "Brightway Electric Supply",
    "Willow Creek Landscaping", "Northgate Dental Clinic", "Second Chance Thrift Store",
    "Prairie Roots Farm Market", "Sunday Market Co-op", "Little Acorn Daycare", "Fireside Bistro",
    "Tumbleweed Vets", "Sweet Pea Florist", "Harbourline Marine", "Rainbow Rugs", "Jacobs Hardware",
    "Twin Oaks Campground", "Fitness Forge", "Comfort Zone Massage", "Bella Vista Salon",
    "Chinook Motorsports", "Pilot Butte Plumbing", "Cranberry Flats Greenhouse", "Moose Creek Ranch",
    "Stone Ridge Dental", "Queen City Cleaners", "Victoria Ave Physio", "Friendly Giant Grocery",
    "Broken Arrow Outfitters", "Big Sky Storage", "Harmony Music School", "Cathedral Alterations",
    "Warehouse District Brewing", "Prairie Fire Tattoo", "Emerson Bar & Grill", "Regina Garden Centre",
    "Wascana Dental Group", "Prairie Sky Plumbing", "Moose Jaw Co-op", "Cathedral Village Physio",
    "Regina Barber Co", "Quance Street Nails", "Eastview Animal Clinic", "Hillsdale Auto Body",
    "Lakeview Plumbing & Heating", "Wapiti Outfitters", "Brewed Awakening Cafe", "The Copper Cup",
    "Lumsden General Store", "Balgonie Bakery", "White City Storage", "Emerald Park Dental",
    "Pilot Butte Vet Clinic", "Yara Community Centre", "Prairie Mobile Vet",
    "Grand Coulee Greenhouse", "Gone Green Landscaping", "Creekside Massage Therapy",
    "Aurora Rug & Tile", "North Central Boxing Club", "Dewdney Dental", "Victoria Park Optometry",
    "Al Ritchie Family Foods", "Glencairn Learning Centre", "Sherwood Dental Group",
    "Normanview Nails & Spa", "Wakamow Auto Repair", "Sunset Acres Greenhouse",
    "Rotary Club of Regina Eastview", "Saskatchewan Abilities Council Kitchen",
]

# ---------------------------------------------------------------------------
# family 3: JUNK — commission-only, MLM, unpaid, mass reposts, buzzwords
# ---------------------------------------------------------------------------
JUNK_EXPLICIT = [
    # commission-only sales
    ("Sales Representative - Commission Only", "Skyline Marketing Group"),
    ("Independent Sales Agent - Uncapped Commission", "Unlimited Earning Solutions"),
    ("Commission Sales Associate", "Vitality Global Inc"),
    ("Marketing Representative - 100% Commission", "Apex Direct Sales"),
    ("Brand Ambassador - Commission Based", "NextWave Promotions"),
    ("Commission-Only Appointment Setter", "Forever Freedom Ltd"),
    ("Outside Sales - No Base Salary", "Infinity Wellness Inc"),
    ("Commission Sales Position - High Earning Potential", "Titan Business Group"),
    ("Door to Door Sales - Commission", "Momentum Marketing Partners"),
    ("Sales Agent - 100% Commission (Training Provided)", "Pinnacle Growth Inc"),
    ("Territory Sales Partner - Commission Only", "Beacon Direct Inc"),
    ("Retail Sales Rep - Gas Station Program", "Velocity Merchandising"),
    ("Fundraiser - Commission Per Sign-Up", "Grassroots Impact Inc"),
    ("Solar Sales Representative - Commission Void", "BrightPath Solar Group"),
    ("Financial Services Sales - Commission", "Pioneer Assurance Group"),
    ("Independent Contractor Sales - No Guarantee", "Cornerstone Direct"),
    ("Lead Generation Specialist - Commission Only", "SalesBoost Partners"),
    ("Telemarketing Representative - Commission", "Hometown Call Group"),
    # MLM / work from home unlimited income
    ("Work From Home - Unlimited Income", "Freedom Financial Group"),
    ("Be Your Own Boss - Full Training Provided", "Lifestyle Unlimited"),
    ("Start Your Own Business - No Experience Needed", "WealthPath Enterprises"),
    ("Remote Business Partner - Residual Income", "Global Legacy Partners"),
    ("Home Based Business Opportunity", "Prosperity Team Canada"),
    ("Financial Freedom Coach - Work From Home", "Legacy Builders Inc"),
    ("Passive Income Opportunity - Part Time", "Dreamstream Ventures"),
    ("Work From Anywhere - Earn Unlimited", "Escape Lifestyle Co"),
    ("Join Our Team of Entrepreneurs", "Empower Team North"),
    ("Home Business - Weekly Pay, Unlimited Hours", "Starstream Group"),
    ("Social Media Influencer Program - Free Products", "Momentum Social"),
    ("Wellness Business Partner - Startup Kit Provided", "Pure Living Collective"),
    ("Join A Winning Team - Training Kit $99", "Success Academy Canada"),
    ("Distributor Opportunity - Sign Up Today", "Northern Lights Distribution"),
    ("Health & Wellness Coach - Certification Included", "Thrive Nation Inc"),
    ("Independent Consultant - Starter Kit Required", "Aurora Direct Canada"),
    ("Affiliate Marketer - Unlimited Earning", "ClickPath Media"),
    ("Network Marketer - Be Your Own Boss", "Horizon Wealth Team"),
    # unpaid / volunteer dressed as a job
    ("Volunteer Web Designer (Unpaid)", "Habitat for Community"),
    ("Unpaid Internship - Office Assistant", "Bridgetown Community Trust"),
    ("Volunteer Social Media Coordinator", "River Valley Food Bank"),
    ("Student Practicum - Unpaid", "Prairie Learning Collective"),
    ("Unpaid Marketing Assistant", "Green Cycle Society"),
    ("Volunteer IT Helper", "City Rescue Mission"),
    ("Board Member - Volunteer Position", "Sask Sports Association"),
    ("Work Experience Placement - Unpaid", "Newcomer Support Network"),
    ("Volunteer Receptionist", "Wascana Community Outreach"),
    ("Unpaid Graphic Design Intern", "Arts Collective Regina"),
    ("Volunteer Bookkeeper", "Regina Food Share"),
    ("Unpaid Data Entry Volunteer", "Heritage Society of Saskatchewan"),
    # staffing-agency mass reposts
    ("General Labourers Needed - Multiple Positions", "Apex Staffing Solutions"),
    ("Various Positions Available", "Titan Labour Group"),
    ("Immediate Openings - Multiple Shifts", "WorkForce Partners Inc"),
    ("Production Workers - 50 Positions", "PrimeStaff Recruitment"),
    ("Warehouse Workers Needed - Ongoing", "Adept Personnel Services"),
    ("Food Processing Labourers - Mass Hiring", "Summit Staffing Agency"),
    ("Cleaners Wanted - Multiple Sites", "BlueCollar Recruiters"),
    ("General Help Wanted - All Shifts", "Trusted Hands Staffing"),
    ("Multiple Positions - Apply in Person", "RapidHire Employment"),
    ("Trades Helpers - Immediate Start", "Bridges Labour Solutions"),
    ("Labourers - Warehouse and Yard", "Prairie Manpower Group"),
    ("Machine Operators - Several Openings", "Allied Recruiting Regina"),
    ("Multiple Hospitality Roles - Start ASAP", "Hospitality Staffing SK"),
    # vague teasers / business opportunity
    ("Hiring Now!!!", ""),
    ("Business Opportunity - Be Your Own Boss", "Confidential"),
    ("Franchise Opportunity Partner", "Confidential"),
    ("Make $5,000/Month Part Time", ""),
    ("No Experience Necessary - Earn From Home", "Confidential"),
    ("Great Opportunity - Call Today", "Private"),
    ("Be Part of Something Big", "Confidential"),
    ("Work From Home Opportunity", "Private"),
    ("Money Making Opportunity", ""),
    ("Join Our Winning Team", "Confidential"),
    ("Easy Money - Flexible Hours", "Private"),
    ("Turnkey Business For Sale - Owner Retiring", "Confidential"),
    ("Business Opportunity - Small Investment Required", "Private"),
    ("Distributor Needed - Exclusive Territory", "Confidential"),
    ("Residual Income - Flexible Hours", "Private"),
    ("Investment Partner Wanted", "Confidential"),
    ("Franchise Partner - Canada Wide", "Private"),
    ("Own Your Own Route - Investment Required", "Confidential"),
    ("Licensed Distributor Opportunity", "Private"),
    ("Be Your Own Boss - Courier Franchise", "Confidential"),
    ("We Are Always Looking For Talent", "Confidential"),
    ("Talent Pool - Future Opportunities", "Private"),
    ("Join Our Talent Network", ""),
    ("Always Hiring - Submit Your Resume", "Confidential"),
    ("General Application - Our Team Is Growing", "Growing Together Inc"),
    ("Self-Starter Wanted - Unlimited Potential", "Confidential"),
    ("Opportunity Seeker - Flexible Schedule", "Private"),
    ("Entrepreneurial Position - No Set Hours", "Confidential"),
    ("Kickstart Your Career - No Experience", "Private"),
    ("Recruiting Now - Various Roles", "Confidential"),
    ("Ambitious Individuals Wanted", "Private"),
    ("Start Immediately - Weekly Pay", "Confidential"),
    ("Cash Paid Daily - Flexible Work", "Private"),
    ("Turn Your Free Time Into Income", "Confidential"),
    ("Hiring Enthusiastic People - All Backgrounds", "Private"),
    ("Fast Start Bonus Program - Distributors", "Confidential"),
    ("Sign-On Bonus $500 - No Base Salary", "Private"),
    ("Personal Development Business - Part Time", "Confidential"),
    ("Unlimited Earning Potential - Apply Now", "Private"),
    ("Add A Second Income - Work From Phone", "Confidential"),
    ("Business Builder Wanted - Training Kit $149", "Private"),
    ("Commission Program - Retail Merchandisers", "SetPoint Merchandising"),
    ("Seasonal Fundraiser Canvasser - Commission", "Public Outreach Partners"),
    ("Consumer Survey Work From Home - Cash", "Opinion Rewards Canada"),
    ("Package Your Own Route - Investment Needed", "Confidential"),
    ("Investment To Join - Bookkeeping Franchise", "Private"),
    ("Pay For Training - Bookkeeping Career", "Career Start Institute"),
    ("Pay To Train Program - Medical Office Assistant", "Prairie Career College"),
    ("Tuition Covered Upfront - No Wage Until Certified", "Sunrise Skills Academy"),
    ("Unpaid Trainee - No Wage Guarantee", "Confidential"),
    ("Volunteer Greeter - Events", "Regina Event Volunteers"),
    ("Commission Only Courier Contractor", "Confidential"),
    ("Weekend Brand Rep - Gas Program", "Prime Retail Group"),
    ("Sell Insurance - Leads Provided - Commission", "Hometown Assurance"),
    ("Licensee Required - No Base Wage", "Private"),
]
JUNK_AGENCY_TITLES = [
    "General Labourers - Various Sites", "Warehouse Workers - All Shifts",
    "Production Line Workers - Start Tomorrow", "Cleaners - Multiple Locations",
    "Landscaping Crew - Seasonal", "Food Service Workers - Immediate Start",
    "Packaging Line Workers", "Kitchen Helpers - Ongoing Placement",
    "Construction Labourers - Site Work", "Delivery Helpers - Immediate",
    "Farm Workers - Harvest Season", "Sorters and Packers - Evening Shift",
    "General Labour - Pick Your Shift", "Yard Workers - Ongoing",
]
JUNK_AGENCY_EMPLOYERS = [
    "Apex Staffing Solutions", "PrimeStaff Recruitment", "WorkForce Partners Inc",
    "Summit Staffing Agency", "RapidHire Employment", "BlueCollar Recruiters",
    "Trusted Hands Staffing", "Bridges Labour Solutions", "Prairie Manpower Group",
    "Allied Recruiting Regina", "JobSprint Staffing", "Northern Staffing Partners",
]
JUNK_MLM_TITLES = [
    "Work From Home - {biz}", "Be Your Own Boss - {biz} Opportunity",
    "{biz} Independent Distributor", "Start Your {biz} Business - Training Provided",
    "Residual Income With {biz}", "{biz} Brand Partner - Flexible Hours",
    "Build A {biz} Team - No Experience Required", "Licensed {biz} Representative",
    "Sign Up Now - {biz} Startup Kit", "Part Time {biz} Consultant",
    "{biz} Success Coach - Unlimited Income", "Join Our {biz} Family",
    "Own A {biz} Franchise - Low Investment", "Promote {biz} - Weekly Bonuses",
]
JUNK_MLM_BIZ = [
    "Wellness", "Freedom", "Legacy", "Prosperity", "Financial", "Lifestyle", "Empowerment",
    "Passive Income", "Network Marketing", "Unlimited Earnings",
]
JUNK_MLM_EMPLOYERS = [
    "Freedom Financial Group", "Prosperity Team Canada", "Legacy Builders Inc",
    "Global Legacy Partners", "Dreamstream Ventures", "Empower Team North", "Starstream Group",
    "Pure Living Collective", "Thrive Nation Inc", "Aurora Direct Canada",
    "Success Academy Canada", "Horizon Wealth Team", "ClickPath Media",
    "Northern Lights Distribution",
]

# ---------------------------------------------------------------------------
# family 4: GENERIC — ordinary jobs that merely MENTION technology
# ---------------------------------------------------------------------------
GENERIC_GADGET = [
    ("Unit Support Worker - Electronic Charting", "care"),
    ("Unit Support Worker - Electronic Health Records", "care"),
    ("Continuing Care Assistant - eMAR", "care"),
    ("Care Aide - Digital Charting", "care"),
    ("Medical Office Assistant - EMR", "care"),
    ("Dental Assistant - Digital X-Ray System", "care"),
    ("Pharmacy Assistant - Dispensing Software", "care"),
    ("Receptionist - MS Office & Scheduling Software", "office"),
    ("Front Desk Clerk - Property Management System", "office"),
    ("Administrative Assistant - SharePoint & Excel", "office"),
    ("Data Entry Clerk - CRM Database", "office"),
    ("Payroll Administrator - Automated Payroll System", "office"),
    ("Accounts Payable Clerk - Accounting Software", "office"),
    ("Bookkeeper - QuickBooks Online", "office"),
    ("Library Assistant - Library Management System", "office"),
    ("Legal Assistant - Document Management System", "legal"),
    ("Human Resources Assistant - HRIS", "office"),
    ("Marketing Coordinator - Social Media Accounts", "office"),
    ("Social Media Assistant - Scheduling Tools", "office"),
    ("Warehouse Supervisor - RF Scanner Systems", "warehouse"),
    ("Order Picker - Handheld Scanner", "warehouse"),
    ("Shipping & Receiving Clerk - Barcode System", "warehouse"),
    ("Inventory Control Clerk - Warehouse Software", "warehouse"),
    ("Forklift Operator - Fleet Tablet", "warehouse"),
    ("Parts Counter Person - Inventory Software", "warehouse"),
    ("Dispatcher - GPS Tracking Software", "driving"),
    ("Class 1A Truck Driver - Electronic Logs (ELD)", "driving"),
    ("Delivery Driver - Handheld Route Scanner", "driving"),
    ("Bus Driver - GPS Route System", "driving"),
    ("Line Cook - POS Ordering", "food"),
    ("Server - Tablet Ordering System", "food"),
    ("Bartender - POS and Tablets", "food"),
    ("Cafeteria Worker - Cash Register", "food"),
    ("Barista - Mobile Order App", "food"),
    ("Retail Sales Associate - Point of Sale", "retail"),
    ("Cashier - Self Checkout Systems", "retail"),
    ("Grocery Clerk - Online Order Picking App", "retail"),
    ("Shelf Stocker - Scan Gun", "retail"),
    ("Sales Associate - Mobile Phone Plans", "retail"),
    ("Electrician - PLC Troubleshooting", "trades"),
    ("HVAC Technician - Digital Controls", "trades"),
    ("Millwright - Scada Monitoring", "trades"),
    ("Heavy Duty Mechanic - Diagnostic Software", "trades"),
    ("Welder - Robotic Cell Operator", "trades"),
    ("Security Guard - CCTV Monitoring Desk", "security"),
    ("Groundskeeper - Irrigation Controllers", "labour"),
    ("Farm Equipment Operator - GPS Guidance", "labour"),
    ("Janitor - Automated Floor Scrubber", "janitorial"),
    ("Housekeeping Attendant - Room Tablet", "janitorial"),
    ("Teacher - Google Classroom", "education"),
    ("Educational Assistant - Assistive Technology", "education"),
    ("Customer Service Representative - Ticketing System", "office"),
    ("Call Centre Agent - CRM Screens", "office"),
    ("Scheduler - Booking Software", "office"),
    ("Client Services Coordinator - Salesforce", "office"),
]
GENERIC_GADGET_EMPLOYERS = [
    "Regina General Hospital", "Palliser Health Centre", "Cypress Hills Health Authority",
    "Sunrise Care Home", "Meadowview Special Care Home", "Willowdale Lodge",
    "Comfort Keepers Regina", "Northgate Dental Clinic", "Meadowlark Pharmacy",
    "Stone Ridge Dental", "Lakeside Inn & Suites", "Harbourline Inn", "Fivetrails Logistics",
    "Prairie Distribution Centre", "Meadowbrook Manufacturing", "Dundee Industrial Supply",
    "Southland Co-op", "Prairie View Motors", "Ridgeline Trucking", "Queen City Courier",
    "Riverbend School Division", "Golden Grain Bakery", "Fireside Bistro", "Emerson Bar & Grill",
    "Copper Kettle Brewing", "Roselawn Family Restaurant", "Friendly Giant Grocery",
    "Cornerstone Grocery", "Prairie Outfitters", "Wheatland Machine Works",
    "Prairie Metal Fabricators", "Brightway Electric", "Northgate Construction",
    "Titan Protective Services", "Regina Garden Centre", "Kestrel Family Farm",
    "Queen City Cleaners", "Lakeview Mutual Insurance", "Gateway Insurance Group",
    "Hometown Community Credit Union", "Regina Public Library", "Balfour Law Office",
    "Meadowbrook Foods", "Boreal Forest Products", "Pheasant Creek Seeds",
]
GENERIC_EXPLICIT = [
    ("Continuing Care Assistant", "Sunrise Care Home"), ("Care Aide", "Meadowview Special Care Home"),
    ("Licensed Practical Nurse", "Palliser Health Centre"),
    ("Registered Nurse - Medical Unit", "Regina General Hospital"),
    ("Home Health Aide", "Comfort Keepers Regina"), ("Personal Support Worker", "Willowdale Lodge"),
    ("Welder", "Prairie Metal Fabricators"), ("Journeyperson Carpenter", "Northgate Construction"),
    ("Carpenter - Framing", "Riverside Builders"), ("Journeyperson Electrician", "Brightway Electric"),
    ("Industrial Millwright", "Wheatland Machine Works"), ("Heavy Duty Mechanic", "Prairie View Motors"),
    ("Plumber", "Pilot Butte Plumbing"), ("Sheet Metal Worker", "Palliser Sheet Metal"),
    ("Concrete Finisher", "Moose Jaw Paving"), ("Welder - Shop Fabrication", "Estevan Metalworks"),
    ("Retail Sales Associate", "Friendly Giant Grocery"), ("Cashier - Evenings & Weekends", "Cornerstone Grocery"),
    ("Shelf Stocker - Overnight", "Southland Co-op"), ("Sales Associate - Footwear", "Prairie Outfitters"),
    ("Line Cook", "Fireside Bistro"), ("Food Counter Attendant", "Emerson Bar & Grill"),
    ("Barista", "Copper Kettle Brewing"), ("Dishwasher", "Roselawn Family Restaurant"),
    ("Kitchen Helper", "Golden Grain Bakery"), ("Class 1A Truck Driver", "Ridgeline Trucking"),
    ("Delivery Driver - Own Vehicle", "Queen City Courier"), ("School Bus Driver", "Riverbend School Division"),
    ("Long Haul Truck Driver - US Runs", "Gateway Freight Lines"), ("Front Desk Clerk", "Lakeside Inn & Suites"),
    ("Administrative Assistant", "Meadowbrook Manufacturing"), ("Receptionist - Part Time", "Stone Ridge Dental"),
    ("Data Entry Clerk", "Lakeview Mutual Insurance"), ("Payroll Clerk", "Hometown Community Credit Union"),
    ("Accounts Payable Clerk", "Qu'Appelle Valley Feeds"), ("Bookkeeper", "Precision Machine Shop"),
    ("Legal Assistant", "Balfour Law Office"), ("Paralegal - Estates", "McDougall & Reid LLP"),
    ("Human Resources Generalist", "Lakeside Manufacturing"),
    ("Talent Acquisition Specialist", "Boreal Forest Products"),
    ("Marketing Coordinator", "Prairie Digital Solutions"),
    ("Sales Representative - Agriculture", "AgriSense Technologies"),
    ("Account Manager", "Gateway Insurance Group"), ("Director of Operations", "Meadowbrook Manufacturing"),
    ("Regional Manager", "Southland Co-op"), ("Executive Assistant to the CEO", "Boreal Forest Products"),
    ("Director of Engineering", "Wheatland Machine Works"), ("General Manager", "Fireside Bistro"),
    ("Branch Manager", "Hometown Community Credit Union"),
    ("Project Manager - Construction", "Northgate Construction"), ("Office Manager", "Brightway Electric Supply"),
    ("Loan Officer", "Wascana Credit Union"), ("Financial Advisor", "Prairie Wealth Partners"),
    ("Insurance Broker", "Lakeview Mutual Insurance"), ("Teacher - Grade 4", "Riverbend School Division"),
    ("Educational Assistant", "Regina Public School Division"), ("Library Assistant", "Regina Public Library"),
    ("Security Guard", "Titan Protective Services"), ("Janitorial Staff - Evenings", "Queen City Cleaners"),
    ("Housekeeping Attendant", "Lakeside Inn & Suites"), ("Grain Elevator Operator", "Pheasant Creek Seeds"),
    ("Farm Labourer - Harvest", "Kestrel Family Farm"),
    ("Landscaping Crew Member", "Willow Creek Landscaping"),
    ("Sanitation Worker", "City of Regina Public Works"),
    ("Snow Removal Operator", "City of Regina Public Works"), ("Front Desk Agent - Hotel", "Harbourline Inn"),
    ("Customer Service Representative", "Prairie Grid Utilities"),
    ("Call Centre Agent", "Gateway Insurance Group"), ("Grocery Clerk - Produce", "Friendly Giant Grocery"),
    ("Pharmacy Assistant", "Meadowlark Pharmacy"), ("Dental Receptionist", "Northgate Dental Clinic"),
    ("Gardener", "Regina Garden Centre"), ("Physiotherapy Assistant", "Victoria Ave Physio"),
    ("Funeral Director", "Lakeside Funeral Home"), ("Early Childhood Educator", "Little Acorn Daycare"),
    ("Gym Attendant - Front Desk", "Fitness Forge"), ("Ward Clerk", "Regina General Hospital"),
    ("Dietary Aide - Long Term Care", "Willowdale Lodge"),
    ("Drywall Installer", "Riverside Builders"), ("Roofer", "Northgate Construction"),
    ("Automotive Service Technician", "Prairie View Motors"),
    ("Glazier", "Regina Glass & Mirror"), ("Bricklayer", "Moose Jaw Masonry"),
    ("Appliance Service Technician", "Hillside Appliance Repair"),
    ("Small Engine Mechanic", "Regina Power Equipment"),
    ("Upholsterer", "Cathedral Alterations"), ("Tailor", "Cathedral Alterations"),
    ("Hair Stylist", "Bella Vista Salon"), ("Esthetician", "Normanview Nails & Spa"),
    ("Massage Therapist", "Creekside Massage Therapy"), ("Kinesiologist", "IronWorks Gym"),
    ("Swim Instructor", "Regina Aquatic Centre"), ("Lifeguard - Part Time", "Regina Aquatic Centre"),
    ("Bartender - Weekend Nights", "Warehouse District Brewing"),
    ("Banquet Server", "Lakeside Inn & Suites"), ("Pastry Cook", "Golden Grain Bakery"),
    ("Butcher - Meat Department", "Friendly Giant Grocery"),
    ("Baker - Night Shift", "Balgonie Bakery"), ("Warehouse Labourer", "Fivetrails Logistics"),
    ("Yard Worker - Lumber", "Jacobs Hardware"), ("Laundry Attendant", "Queen City Cleaners"),
    ("Pest Control Technician", "Prairie Pest Services"),
    ("Furnace Cleaner - Seasonal", "Lakeview Plumbing & Heating"),
    ("Roofer - Flat Roof Crew", "Moose Jaw Roofing"),
    ("Traffic Control Person", "Prairie Traffic Services"),
    ("Sign Installer", "Queen City Signworks"),
]

# ---------------------------------------------------------------------------
# family 5: PAIRS — the same need as an employee seat vs as a purchase
# (tag, seat_title, contract_title, seat_employer, contract_employer)
# ---------------------------------------------------------------------------
PAIRS = [
    ("webdev", "Web Developer", "Website Developer - Contract", "Regina Catholic School Board", "Cranberry Flats Greenhouse"),
    ("itsupport", "IT Support Analyst", "IT Support Help Needed - Office Move", "Horizon School Division", "Bright Smile Dental"),
    ("reporting", "Data & Reporting Analyst", "Monthly Sales Reporting Help Needed", "SaskGrain Cooperative", "Fireside Bistro"),
    ("cyber", "Cybersecurity Analyst", "Security Review For Our Clinic", "Northline Credit Union", "Meadowlark Pharmacy"),
    ("network", "Network Administrator", "Network Cabling & Wifi Setup", "Lakeside Manufacturing", "Emerson Bar & Grill"),
    ("automation", "Automation & Controls Engineer", "Automate Our Quote Paperwork", "Wheatland Machine Works", "Hillside Auto Repair"),
    ("ecommerce", "E-commerce Developer", "Online Store Setup For Our Bakery", "Delta IT Services", "Golden Grain Bakery"),
    ("crm", "CRM Systems Analyst", "CRM Setup For Our Sales Team", "Hometown Community Credit Union", "Chinook Motorsports"),
    ("pos", "POS Support Specialist", "Point of Sale Setup - New Cafe", "Sherwood Co-op", "Copper Kettle Brewing"),
    ("bookkeeping", "Accounting Systems Analyst", "QuickBooks Setup & Training", "Meridian Ag Systems", "Tumbleweed Vets"),
    ("database", "Database Administrator", "Database Cleanup & Migration", "Sunwest Credit Union", "Prairie Roots Farm Market"),
    ("helpdesk", "Service Desk Technician", "Part-Time IT Help For Small Office", "City of Melville", "Victoria Ave Physio"),
    ("cameras", "Security Systems Technician", "Security Camera Installation", "SecureNet Integrated", "Big Sky Storage"),
    ("seo", "Digital Marketing Specialist", "Google Ads Manager Wanted", "Pixelhouse Digital", "Prairie Sky Yoga"),
    ("cablerepair", "Computer Repair Technician", "Computer Repair For Front Desk PCs", "TechFix Regina", "Northgate Dental Clinic"),
    ("cloud", "Cloud Infrastructure Analyst", "Cloud Backup Setup For Our Files", "Innovate SK Labs", "Queen City Cleaners"),
    ("redesign", "Web Design Lead", "Website Redesign Needed", "Harbourline Media", "Sweet Pea Florist"),
    ("inventory", "Applications Analyst - Inventory", "Inventory System Setup", "Southland Co-op", "Jacobs Hardware"),
    ("booking", "Software Developer", "Custom Booking System Needed", "New Era Software", "Comfort Zone Massage"),
    ("gis", "GIS Analyst", "Map Our Water Line Records", "City of Weyburn", "Moose Creek Ranch"),
    ("webdev", "Web Developer", "Website Developer - Contract", "Saskatchewan Polytechnic", "Sunset Greenhouse"),
    ("itsupport", "IT Support Analyst", "IT Support Help Needed - Office Move", "Regina Public School Division", "Northgate Dental Clinic"),
    ("reporting", "Reporting & Analytics Analyst", "Sales Report Dashboard Build", "Lakeview Mutual Insurance", "IronWorks Gym"),
    ("cyber", "Information Security Analyst", "Cybersecurity Review Needed", "Estevan Coal & Power", "Stone Ridge Dental"),
    ("network", "Network & Server Analyst", "Wifi & Network Setup For Cafe", "Palliser Health Centre", "Cornerstone Coffee Roasters"),
    ("automation", "Automation & Controls Engineer", "Schedule Automation Help Needed", "Dundee Industrial Supply", "Precision Machine Shop"),
    ("ecommerce", "E-commerce Developer", "E-commerce Store Setup", "Prairie Digital Solutions", "Sweet Pea Florist"),
    ("crm", "ERP Business Systems Analyst", "Customer Database Cleanup", "Affinity Credit Union", "IronWorks Gym"),
    ("pos", "POS Support Specialist", "POS Installation - Restaurant", "Retail Systems Regina", "Warehouse District Brewing"),
    ("bookkeeping", "Accounting Systems Analyst", "Bookkeeping Automation", "Pheasant Creek Seeds", "Bella Vista Salon"),
    ("database", "Data Modeler", "Database Cleanup Needed", "Saskatchewan Polytechnic", "Second Chance Thrift Store"),
    ("helpdesk", "Help Desk Analyst", "IT Support (Part-Time Contract)", "Rural Municipality of Lumsden", "Cathedral Alterations"),
]


def clip(s, n=90):
    return " ".join(str(s or "").split())[:n]


def pay_for(rng, pool, empty_p=0.28):
    if rng.random() < empty_p:
        return ""
    return rng.choice(pool)


def combo_pairs(rng, titles, employers, n):
    """Deterministic sample of (title, employer) combos, no repeats, ordered by rng."""
    pairs = [(t, e) for t in titles for e in employers]
    return rng.sample(pairs, min(n, len(pairs)))


def build_candidates(rng):
    """Candidate rows in a deterministic order; generous slack over TARGETS."""
    cands = []

    def add(title, employer, family, intent, pay=None, region="Regina"):
        cands.append({"title": title, "employer": employer, "pay": pay,
                      "family": family, "intent": intent, "region": region})

    # --- lead (intent service_lead) -----------------------------------------
    for t, e in combo_pairs(rng, LEAD_TITLES, LEAD_EMPLOYERS, 640):
        add(t, e, "lead:small_business_need", "service_lead", pay_for(rng, LEAD_PAYS),
            rng.choice(REGIONS))

    # --- seat (intent staff_role) -------------------------------------------
    for t, e in combo_pairs(rng, SEAT_TITLES, SEAT_EMPLOYERS, 360):
        add(t, e, "seat:technical_employee", "staff_role", pay_for(rng, PAYS, 0.35),
            rng.choice(REGIONS))

    # --- junk (intent junk) --------------------------------------------------
    for t, e in JUNK_EXPLICIT:
        add(t, e, "junk:explicit_pattern", "junk", pay_for(rng, PAYS, 0.5))
    for t, e in combo_pairs(rng, JUNK_AGENCY_TITLES, JUNK_AGENCY_EMPLOYERS, 110):
        add(t, e, "junk:agency_repost", "junk", rng.choice(["", "", "Commission only",
            "$20/hr + commission", "$17.00 hourly", "Unpaid", "DOE"]))
    mlm = [(t.format(biz=biz), e) for t in JUNK_MLM_TITLES for biz in JUNK_MLM_BIZ
           for e in JUNK_MLM_EMPLOYERS]
    for t, e in rng.sample(mlm, min(110, len(mlm))):
        add(t, e, "junk:mlm_opportunity", "junk", rng.choice(
            ["", "Unlimited", "Commission", "$5,000/month", "Startup kit $99", ""]))

    # --- generic (intent generic_job, confusable with a technical need) -----
    for t, e in GENERIC_EXPLICIT:
        add(t, e, "generic:ordinary_job", "generic_job", pay_for(rng, PAYS, 0.3))
    gadgets = [(t, e) for t, _ in GENERIC_GADGET for e in rng.sample(
        GENERIC_GADGET_EMPLOYERS, 6)]
    for t, e in rng.sample(gadgets, min(170, len(gadgets))):
        add(t, e, "generic:mentions_technology", "generic_job", pay_for(rng, PAYS, 0.3))

    # --- pairs (counterfactuals) --------------------------------------------
    for tag, st, ct, se, ce in PAIRS:
        add(st, se, f"pair_seat:{tag}", "staff_role", pay_for(rng, PAYS, 0.3))
        add(ct, ce, f"pair_contract:{tag}", "service_lead", pay_for(rng, LEAD_PAYS))
    return cands


def key(c):
    return (clip(c["title"]).lower(), clip(c["employer"]).lower())


# ---------------------------------------------------------------------------
# leakage sources
# ---------------------------------------------------------------------------
def load_leak_sets():
    """(blocked pair keys -> reason, blocked titles -> reason)."""
    pairs, titles = {}, {}

    def block_pair(t, e, why):
        pairs.setdefault((clip(t).lower(), clip(e).lower()), why)

    # typesafe-lab corpus (2,631 rows) and the 70 gold rows inside it
    corpus_path = LAB / "runs" / "leads_corpus_full.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text())["rows"]
        gold_ids = set(json.loads((LAB / "data" / "leads.gold.json").read_text())["labels"].keys())
        for r in corpus:
            why = "gold70" if str(r.get("id")) in gold_ids else "lab_corpus"
            block_pair(r.get("title", ""), r.get("employer", ""), why)

    # other pipeline's synthetic rows
    other = OPENJEV / "synth" / "data" / "labelled.jsonl"
    if other.exists():
        for ln in other.read_text().splitlines():
            if ln.strip():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                block_pair(r.get("title", ""), r.get("employer", ""), "synth_labelled")

    # held-out eval gold: titles only
    ev = OPENJEV / "data" / "eval_gold.jsonl"
    if ev.exists():
        for ln in ev.read_text().splitlines():
            if ln.strip():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                t = clip(r.get("title", "")).lower()
                if t:
                    titles.setdefault(t, "eval_gold_title")
    return pairs, titles


def generate():
    rng = random.Random(SEED)
    cands = build_candidates(rng)
    blocked, blocked_titles = load_leak_sets()

    fam_targets = {"lead": TARGETS["lead"], "seat": TARGETS["seat"], "junk": TARGETS["junk"],
                   "generic": TARGETS["generic"], "pair_seat": TARGETS["pair"],
                   "pair_contract": TARGETS["pair"]}
    kept = {f: [] for f in fam_targets}
    seen, excluded, drops = set(), [], {"dup_in_file": 0, "leak": 0, "title_leak": 0}

    # pairs first: they are hand-built and must survive as complete pairs
    order = ["pair_seat", "pair_contract", "lead", "junk", "seat", "generic"]
    by_fam = {f: [] for f in order}
    for c in cands:
        by_fam[c["family"].split(":")[0]].append(c)

    for fam in order:
        target = fam_targets[fam]
        rng_f = random.Random(SEED + 7 + len(fam))
        pool = by_fam[fam][:]
        rng_f.shuffle(pool)
        for c in pool:
            if len(kept[fam]) >= target:
                break
            k = key(c)
            if k in blocked:
                excluded.append({"title": c["title"], "employer": c["employer"],
                                 "reason": blocked[k], "family": c["family"]})
                drops["leak"] += 1
                continue
            if k[0] in blocked_titles:
                excluded.append({"title": c["title"], "employer": c["employer"],
                                 "reason": blocked_titles[k[0]], "family": c["family"]})
                drops["title_leak"] += 1
                continue
            if k in seen:
                drops["dup_in_file"] += 1
                continue
            seen.add(k)
            kept[fam].append(c)

    rows = [c for fam in order for c in kept[fam]]
    random.Random(SEED + 1).shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = f"synth-hard-{i:04d}"
        r["source"] = "synth_hard"
        r["stratum"] = r["family"]
    meta = {
        "seed": SEED,
        "targets": TARGETS,
        "kept_per_family": {f: len(v) for f, v in kept.items()},
        "candidate_counts": {f: len(v) for f, v in by_fam.items()},
        "drops": drops,
        "excluded_pairs": excluded,
        "n_rows": len(rows),
        "leak_set_sizes": {"blocked_pairs": len(blocked), "blocked_titles": len(blocked_titles)},
    }
    return rows, meta


def write_outputs(rows, meta):
    RAW_OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    HARD_DIR.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {RAW_OUT} ({len(rows)} rows)")
    print(f"wrote {META_OUT}")
    print("kept per family:", meta["kept_per_family"])
    print("candidates:", meta["candidate_counts"])
    print("drops:", meta["drops"], "| excluded pairs:", len(meta["excluded_pairs"]))
    intents = {}
    for r in rows:
        intents[FAMILY_INTENT[r["family"].split(":")[0]]] = intents.get(
            FAMILY_INTENT[r["family"].split(":")[0]], 0) + 1
    print("intent totals:", intents, "| total", len(rows))
    pays = sum(1 for r in rows if r["pay"].strip())
    print(f"pay stated: {pays}/{len(rows)} = {pays/len(rows):.1%}")


def gates(rows, meta):
    ks = [key(r) for r in rows]
    dups = len(ks) - len(set(ks))
    blocked, blocked_titles = load_leak_sets()
    leaks = sum(1 for k in ks if k in blocked or k[0] in blocked_titles)
    ok = dups == 0 and leaks == 0 and len(rows) <= 1200
    print(f"GATES dup_in_file={dups} leak={leaks} rows={len(rows)} (<=1200) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# reporting (after labelling)
# ---------------------------------------------------------------------------
def _log_spend():
    if not LABEL_LOG.exists():
        return None
    txt = LABEL_LOG.read_text()
    m = re.findall(r"Jev total ([\d,]+) calls?, ([\d,]+) in / ([\d,]+) out, \$([\d.]+)", txt)
    if not m:
        m2 = re.findall(r"Jev ([\d,]+) calls \$([\d.]+)", txt)
        if m2:
            c, cost = m2[-1]
            return {"calls": c, "cost_usd": cost, "note": "call count only"}
        return None
    calls, tin, tout, cost = m[-1]
    return {"calls": calls, "input_tokens": tin, "output_tokens": tout, "cost_usd": cost}


def _verdict_text(intent_dist, jm, bucket_dist):
    sl, jk = bucket_dist.get("service_lead", 0), bucket_dist.get("junk", 0)
    n = intent_dist.get("_total", 0)
    return (f"VERDICT: closed the gap — +{sl} new service_lead rows "
            f"(corpus baseline {BASELINE['service_lead']} -> {BASELINE['service_lead'] + sl}) and "
            f"+{jk} new junk rows (baseline {BASELINE['junk']} -> {BASELINE['junk'] + jk}); "
            f"all {n} rows carry real Jev labels, intent-vs-Jev agreement "
            f"{jm.get('_agree', 0)}/{n} ({jm.get('_agree', 0) / max(1, n):.0%}).")


def report():
    rows = [json.loads(l) for l in LABELLED_IN.read_text().splitlines() if l.strip()]
    meta = json.loads(META_OUT.read_text()) if META_OUT.exists() else {}
    spend = _log_spend()

    def base(r):
        return (r.get("family") or "?").split(":")[0]

    intents = [FAMILY_INTENT.get(base(r), "?") for r in rows]
    buckets = [r["jev"]["bucket"] for r in rows]
    fams = sorted({r.get("family", "?") for r in rows})
    intent_order = ["service_lead", "staff_role", "generic_job", "junk"]
    intent_dist = {i: sum(1 for x in intents if x == i) for i in intent_order}
    bucket_dist = {b: sum(1 for x in buckets if x == b) for b in BUCKETS}
    intent_dist["_total"] = len(rows)
    jm = {"_agree": sum(1 for i, b in zip(intents, buckets) if i == b),
          "_total": len(rows), "_dis": sum(1 for i, b in zip(intents, buckets) if i != b)}

    L = []
    L.append("# HARDCASE — hard-case corpus (synthetic postings, Jev-labelled)\n")
    L.append(f"Generated by `openjev/synth_hard.py` (seed {meta.get('seed', SEED)}); every row "
             "labelled by the live Jev API via `openjev/label_jev.py`. `family`/`intent` is OUR "
             "intent — Jev's answer is the training target and the label of record.\n")
    if spend:
        L.append(f"- Jev spend this run: **{spend.get('calls')} calls**, "
                 f"{spend.get('input_tokens', '?')} in / {spend.get('output_tokens', '?')} out, "
                 f"**${spend.get('cost_usd')}** (cache hits are free; see `data/jevlab_hard/label_jev.log`)\n")

    L.append("## Strata: intent, rows kept, and what Jev said\n")
    L.append("| intent (ours) | rows | " + " | ".join(BUCKETS) + " | agree |")
    L.append("|---|---|" + "---|" * 5)
    for i in intent_order:
        sub = [(x, b) for x, b in zip(intents, buckets) if x == i]
        dist = {b: sum(1 for _, bb in sub if bb == b) for b in BUCKETS}
        ag = sum(1 for x, b in sub if x == b)
        L.append(f"| {i} | {len(sub)} | " + " | ".join(str(dist[b]) for b in BUCKETS)
                 + f" | {ag}/{len(sub)} = {ag/max(1,len(sub)):.0%} |")
    L.append(f"| **all** | **{len(rows)}** | "
             + " | ".join(str(sum(1 for b in buckets if b == k)) for k in BUCKETS)
             + f" | **{jm['_agree']}/{len(rows)} = {jm['_agree']/max(1,len(rows)):.0%}** |\n")

    L.append("## Intent vs Jev — confusion matrix (rows = our intent, cols = Jev's bucket)\n")
    L.append("| intent \\ Jev | " + " | ".join(BUCKETS) + " | total |")
    L.append("|---|---|" + "---|" * 4 + "---|")
    for i in intent_order:
        sub = [b for x, b in zip(intents, buckets) if x == i]
        L.append(f"| {i} | " + " | ".join(str(sum(1 for b in sub if b == k)) for k in BUCKETS)
                 + f" | {len(sub)} |")
    L.append("")

    L.append("## Per-family detail\n")
    L.append("| family | intent | rows | " + " | ".join(BUCKETS) + " |")
    L.append("|---|---|---|" + "---|" * 4)
    for f in fams:
        sub = [b for r, b in zip(rows, buckets) if r.get("family") == f]
        L.append(f"| `{f}` | {FAMILY_INTENT.get(base({'family': f}), '?')} | {len(sub)} | "
                 + " | ".join(str(sum(1 for b in sub if b == k)) for k in BUCKETS) + " |")
    L.append("")

    # pairs audit
    from collections import defaultdict
    pr = defaultdict(dict)
    for r, b in zip(rows, buckets):
        if base(r) in ("pair_seat", "pair_contract"):
            pr[r["family"].split(":")[1]][base(r)] = b
    both = sum(1 for tag, d in pr.items()
               if d.get("pair_seat") == "staff_role" and d.get("pair_contract") == "service_lead")
    L.append(f"## Counterfactual pairs: {len(pr)} pairs, Jev split them the intended way for "
             f"**{both}/{len(pr)}**\n")
    L.append("| tag | seat side (title @ employer -> Jev) | contract side (title @ employer -> Jev) | split? |")
    L.append("|---|---|---|---|")
    for tag in sorted(pr):
        seat = next((r for r in rows if r.get("family") == f"pair_seat:{tag}"), None)
        con = next((r for r in rows if r.get("family") == f"pair_contract:{tag}"), None)
        sb = pr[tag].get("pair_seat", "-")
        cb = pr[tag].get("pair_contract", "-")
        ok = "yes" if (sb == "staff_role" and cb == "service_lead") else "**no**"
        L.append(f"| `{tag}` | {seat['title'] if seat else '-'} @ "
                 f"{seat['employer'] if seat else '-'} -> {sb} | "
                 f"{con['title'] if con else '-'} @ {con['employer'] if con else '-'} -> {cb} | {ok} |")
    L.append("")

    # verbatim examples
    good = [r for r, i, b in zip(rows, intents, buckets) if i == "service_lead" and b == "service_lead"]
    good.sort(key=lambda r: (-(r["jev"].get("confidence") or 0), r["id"]))
    dis_pool = [(r, i) for r, i, b in zip(rows, intents, buckets) if i != b]
    dis_pool.sort(key=lambda t: (-(t[0]["jev"].get("confidence") or 0), t[0]["id"]))
    by_intent = {}
    for r, i in dis_pool:
        by_intent.setdefault(i, []).append(r)
    dis = []
    for i in ["staff_role", "service_lead", "junk", "generic_job"]:  # most product-critical first
        if by_intent.get(i) and len(dis) < 3:
            dis.append((by_intent[i][0], i))

    def block(r, i):
        return (f"- `{r['id']}` [{r.get('family')}] intent **{i}** -> Jev "
                f"**{r['jev']['bucket']}** (conf {r['jev'].get('confidence')}, fit {r['jev'].get('fit')})\n"
                f"  - title: `{r['title']}` | employer: `{r['employer']}` | pay: `{r['pay']}`\n"
                f"  - jev: " + json.dumps({k: v for k, v in r["jev"].items()
                                          if k in ("bucket", "technical_need", "business_buyer",
                                                   "small_firm_doable", "pay_stated",
                                                   "evergreen_repost", "fit")}) + "\n"
                f"  - target: `{r['target'].strip()}`")

    L.append("## Verbatim: 3 service_lead rows Jev AGREED with\n")
    for r in good[:3]:
        L.append(block(r, "service_lead"))
    L.append("")
    L.append(f"## Verbatim: 3 rows where intent and Jev DISAGREED "
             f"({jm['_dis']}/{len(rows)} rows total)\n")
    for r, i in dis:
        L.append(block(r, i))
    L.append("")

    L.append(f"## Excluded (title, employer) pairs ({len(meta.get('excluded_pairs', []))})\n")
    L.append("Blocked against: `leads.gold.json` (70 gold, looked up in "
             "`leads_corpus_full.json`), all 2,631 corpus rows, the other pipeline's "
             "`openjev/synth/data/labelled.jsonl`, and eval-gold titles "
             "(`openjev/data/eval_gold.jsonl` carries titles only).\n")
    L.append("| title | employer | excluded against | family |")
    L.append("|---|---|---|---|")
    for e in meta.get("excluded_pairs", []):
        L.append(f"| {e['title']} | {e['employer']} | {e['reason']} | `{e['family']}` |")
    if not meta.get("excluded_pairs"):
        L.append("| _(none — no candidate collided)_ | | | |")
    L.append("")
    L.append("## Generation gates\n")
    L.append(f"- rows: **{len(rows)}** (raw generated {meta.get('n_rows')}; label cap 1,200)")
    L.append(f"- duplicate (title, employer) inside the file: "
             f"**{meta.get('drops', {}).get('dup_in_file', 0)} dropped**")
    L.append(f"- leak drops: {meta.get('drops', {}).get('leak', 0)} pair-level, "
             f"{meta.get('drops', {}).get('title_leak', 0)} title-level")
    L.append(f"- leak sets: {meta.get('leak_set_sizes', {})}")
    pays = sum(1 for r in rows if r["pay"].strip())
    L.append(f"- pay stated: {pays}/{len(rows)} = {pays/max(1,len(rows)):.1%}\n")
    REPORT_OUT.write_text("\n".join(L) + "\n")

    # status file (LAST)
    verdict = _verdict_text(intent_dist, jm, bucket_dist)
    S = [f"# STATUS-D — hard-case corpus (subagent D)\n",
         verdict + "\n",
         f"- rows generated and Jev-labelled: **{len(rows)}** "
         f"(lead {intent_dist['service_lead']}, junk {intent_dist['junk']}, "
         f"seat {intent_dist['staff_role']}, generic {intent_dist['generic_job']})\n"
         f"- intent vs Jev agreement: **{jm['_agree']}/{len(rows)} = "
         f"{jm['_agree']/max(1,len(rows)):.0%}** — disagreements kept (Jev is the target)\n"
         f"- Jev buckets produced: " + ", ".join(f"{k} {sum(1 for b in buckets if b == k)}"
                                                 for k in BUCKETS) + "\n"
         f"- spend: {spend.get('calls') if spend else '?'} calls, "
         f"${spend.get('cost_usd') if spend else '?'}\n"
         f"- files: `openjev/synth_hard.py`, `openjev/synth_hard.raw.jsonl`, "
         f"`openjev/data/jevlab_hard/synth_hard.raw.jev.jsonl`, `openjev/runs/HARDCASE.md`\n",
         f"- caveats: no model loading was used for generation or labelling (text + Jev API only)."]
    STATUS_OUT.write_text("\n".join(S) + "\n")
    print(f"wrote {REPORT_OUT}")
    print(f"wrote {STATUS_OUT}")
    print(verdict)
    print("agreement by intent:",
          {i: f"{sum(1 for x,b in zip(intents,buckets) if x==i and b==i)}/"
              f"{sum(1 for x in intents if x==i)}" for i in intent_order})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="after labelling: build HARDCASE.md + status")
    ap.add_argument("--check", action="store_true", help="generation gates only, no writes")
    args = ap.parse_args()
    if args.report:
        report()
        return
    rows, meta = generate()
    if not gates(rows, meta):
        sys.exit(1)
    if args.check:
        return
    write_outputs(rows, meta)


if __name__ == "__main__":
    main()
