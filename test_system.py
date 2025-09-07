#!/usr/bin/env python3
"""
Test Script for Resume ATS Analyzer
Demonstrates the system functionality with sample job descriptions
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from job_analyzer import JobAnalyzer
from matcher import ResumeJobMatcher

def test_job_analyzer():
    """Test the job analyzer with sample job descriptions"""
    
    print("🧪 Testing Job Analyzer...")
    print("=" * 50)
    
    analyzer = JobAnalyzer()
    
    # Test with sample job description
    job_file = "data/sample_jobs/software_engineer_ml.txt"
    
    if not os.path.exists(job_file):
        print(f"❌ Sample job file not found: {job_file}")
        return
    
    with open(job_file, 'r', encoding='utf-8') as f:
        job_text = f.read()
    
    try:
        result = analyzer.analyze_job_description(job_text, "Software Engineer - ML")
        
        print("✅ Job description analyzed successfully!")
        print(f"📋 Job Title: {result['job_title']}")
        print(f"🛠️ Required Skills Found: {len(analyzer.get_all_required_skills_flat(result['required_skills']))}")
        print(f"🎓 Education Requirements: {result['education_requirements']['required_degrees']}")
        print(f"💼 Experience Required: {result['experience_requirements']['min_years']} years")
        print(f"🏢 Company: {result['company_info']['company_name']}")
        print(f"📍 Location: {result['company_info']['location']}")
        
        print("\nSkills by category:")
        for category, skills in result['required_skills'].items():
            print(f"  {category}: {', '.join(skills)}")
        
        print(f"\nMust-have requirements: {len(result['requirements_categorized']['must_have'])}")
        print(f"Nice-to-have requirements: {len(result['requirements_categorized']['nice_to_have'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing job analyzer: {e}")
        return None

def test_complete_workflow_simulation():
    """Simulate a complete workflow without actual resume PDFs"""
    
    print("\n\n🔄 Testing Complete Workflow (Simulated)...")
    print("=" * 50)
    
    # Create simulated resume data
    simulated_resume_data = {
        'file_path': 'simulated_resume.pdf',
        'cleaned_text': """John Smith Software Engineer
        
        Experience: 4 years of software development experience at TechStart Inc.
        Worked on machine learning projects using Python and TensorFlow.
        
        Skills: Python, Java, TensorFlow, scikit-learn, SQL, Git, AWS, Docker
        Experience with data analysis and visualization.
        
        Education: Bachelor of Science in Computer Science from University of Technology
        
        Projects:
        - Built a recommendation system using collaborative filtering
        - Developed a web application using React and Node.js
        - Created data pipelines for real-time analytics
        """,
        'skills': {
            'programming_languages': ['Python', 'Java'],
            'data_science': ['Tensorflow', 'Scikit-Learn'],
            'databases': ['Sql'],
            'cloud_platforms': ['Aws', 'Docker'],
            'tools': ['Git']
        },
        'experience': {
            'total_years': 4,
            'companies': ['TechStart Inc'],
            'positions': [],
            'experience_text': '4 years of software development experience'
        },
        'education': {
            'degrees': ['bachelor of science in computer science'],
            'institutions': ['University of Technology'],
            'fields_of_study': ['Computer Science']
        },
        'projects': [
            'Built a recommendation system using collaborative filtering',
            'Developed a web application using React and Node.js',
            'Created data pipelines for real-time analytics'
        ],
        'contact_info': {
            'email': 'john.smith@email.com',
            'phone': '555-123-4567',
            'linkedin': 'linkedin.com/in/johnsmith',
            'github': 'github.com/johnsmith'
        },
        'word_count': 89,
        'parsed_successfully': True
    }
    
    # Load job description
    job_file = "data/sample_jobs/software_engineer_ml.txt"
    
    if not os.path.exists(job_file):
        print(f"❌ Sample job file not found: {job_file}")
        return
    
    with open(job_file, 'r', encoding='utf-8') as f:
        job_text = f.read()
    
    try:
        # Analyze job
        job_analyzer = JobAnalyzer()
        job_data = job_analyzer.analyze_job_description(job_text, "Software Engineer - ML")
        
        # Match resume to job
        matcher = ResumeJobMatcher()
        match_results = matcher.match_resume_to_job(simulated_resume_data, job_data)
        
        # Display results
        print("✅ Complete workflow test successful!")
        print("\n" + matcher.format_match_results(match_results))
        
        return match_results
        
    except Exception as e:
        print(f"❌ Error in complete workflow test: {e}")
        return None

def show_expected_output_format():
    """Show what kind of output users can expect"""
    
    print("\n\n📋 Expected Output Format:")
    print("=" * 50)
    
    example_output = """
==================================================
RESUME ANALYSIS RESULTS
==================================================

Overall Match Score: 78.5% (Good Match)
Job Title: Software Engineer - Machine Learning

SKILL ANALYSIS:
✅ Found Skills: Python, Java, Tensorflow, Git, Aws
❌ Missing Skills: PyTorch, Kubernetes, React
📊 Skill Match: 71.4%

EXPERIENCE ANALYSIS:
✅ Resume: 4 years, Required: 3+ years
📊 Experience Match: 100.0%

EDUCATION ANALYSIS:
✅ Found 1 degree matches out of 1 requirements
📊 Education Match: 100.0%

ADDITIONAL METRICS:
📝 Text Similarity: 65.2%
🔍 Keyword Match: 58.7%

RECOMMENDATIONS:
1. Consider adding these skills: PyTorch, Kubernetes, React
2. Highlight relevant projects and accomplishments
    """
    
    print(example_output)

def main():
    """Main test function"""
    
    print("🚀 Resume ATS Analyzer - System Test")
    print("=" * 60)
    print("This script demonstrates the system's capabilities")
    print("=" * 60)
    
    # Test 1: Job Analyzer
    job_result = test_job_analyzer()
    
    # Test 2: Complete Workflow Simulation
    if job_result:
        workflow_result = test_complete_workflow_simulation()
    
    # Test 3: Show expected output format
    show_expected_output_format()
    
    print("\n" + "=" * 60)
    print("🎯 SYSTEM CAPABILITIES DEMONSTRATED")
    print("=" * 60)
    print("✅ PDF resume parsing (text extraction, skill detection)")
    print("✅ Job description analysis (requirements extraction)")
    print("✅ Smart matching algorithm (weighted scoring)")
    print("✅ Detailed compatibility reports")
    print("✅ Skills gap analysis and recommendations")
    print("✅ Batch processing capabilities")
    print("✅ Interactive and command-line interfaces")
    
    print("\n📖 To use the system with your own resumes:")
    print("1. Add your resume PDFs to data/sample_resumes/")
    print("2. Run: python main.py --interactive")
    print("3. Follow the prompts to analyze your resumes")
    
    print("\n🔧 Command line usage:")
    print("python main.py --resume your_resume.pdf --job job_description.txt")
    
    print("\n📊 The system provides:")
    print("• Overall compatibility score (0-100%)")
    print("• Detailed skill analysis with gap identification")
    print("• Experience and education requirement matching")
    print("• Semantic similarity analysis")
    print("• Actionable recommendations for improvement")


if __name__ == "__main__":
    main()
