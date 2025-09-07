#!/usr/bin/env python3
"""
Resume ATS Analyzer - Main Application
A comprehensive system for analyzing resumes against job descriptions using NLP and ML
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from resume_parser import ResumeParser
from job_analyzer import JobAnalyzer  
from matcher import ResumeJobMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ATSAnalyzer:
    def __init__(self):
        """Initialize the ATS Analyzer with all components"""
        logger.info("Initializing ATS Analyzer...")
        
        try:
            self.resume_parser = ResumeParser()
            self.job_analyzer = JobAnalyzer()
            self.matcher = ResumeJobMatcher()
            logger.info("ATS Analyzer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize ATS Analyzer: {e}")
            raise

    def analyze_single_resume(self, resume_path: str, job_text: str, 
                            job_title: str = "Not Specified") -> Dict:
        """Analyze a single resume against a job description"""
        
        logger.info(f"Starting analysis for resume: {resume_path}")
        
        try:
            # Parse resume
            logger.info("Parsing resume...")
            resume_data = self.resume_parser.parse_resume(resume_path)
            
            # Analyze job description
            logger.info("Analyzing job description...")
            job_data = self.job_analyzer.analyze_job_description(job_text, job_title)
            
            # Match resume to job
            logger.info("Matching resume to job...")
            match_results = self.matcher.match_resume_to_job(resume_data, job_data)
            
            logger.info("Analysis completed successfully!")
            return match_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    def analyze_multiple_resumes(self, resume_folder: str, job_text: str, 
                               job_title: str = "Not Specified") -> List[Dict]:
        """Analyze multiple resumes in a folder against a job description"""
        
        logger.info(f"Starting batch analysis for resumes in: {resume_folder}")
        
        # Find all PDF files in the folder
        resume_folder_path = Path(resume_folder)
        if not resume_folder_path.exists():
            raise FileNotFoundError(f"Resume folder not found: {resume_folder}")
        
        pdf_files = list(resume_folder_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {resume_folder}")
            return []
        
        results = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Analyzing resume: {pdf_file.name}")
                result = self.analyze_single_resume(str(pdf_file), job_text, job_title)
                result['resume_filename'] = pdf_file.name
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {pdf_file.name}: {e}")
                continue
        
        # Sort results by overall score (highest first)
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        logger.info(f"Batch analysis completed. Analyzed {len(results)} resumes.")
        return results

    def save_results_to_file(self, results: Dict or List[Dict], 
                           output_file: str = "analysis_results.json"):
        """Save analysis results to a JSON file"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def interactive_mode(self):
        """Run the analyzer in interactive mode"""
        
        print("\\n" + "="*60)
        print("🚀 Welcome to Resume ATS Analyzer!")
        print("="*60)
        print()
        print("This tool will analyze your resume(s) against job descriptions")
        print("and provide detailed matching scores and recommendations.")
        print()
        
        while True:
            print("\\nChoose an option:")
            print("1. Analyze single resume")
            print("2. Analyze multiple resumes (batch mode)")
            print("3. Exit")
            
            choice = input("\\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                self._interactive_single_analysis()
            elif choice == '2':
                self._interactive_batch_analysis()
            elif choice == '3':
                print("\\nThank you for using Resume ATS Analyzer!")
                break
            else:
                print("\\n❌ Invalid choice. Please select 1, 2, or 3.")

    def _interactive_single_analysis(self):
        """Handle single resume analysis in interactive mode"""
        
        print("\\n" + "-"*40)
        print("📄 Single Resume Analysis")
        print("-"*40)
        
        # Get resume file path
        resume_path = input("\\nEnter the path to your resume PDF: ").strip().strip('"')
        
        if not os.path.exists(resume_path):
            print(f"\\n❌ Resume file not found: {resume_path}")
            return
        
        # Get job information
        print("\\nHow would you like to provide the job description?")
        print("1. Enter job description text directly")
        print("2. Load from a text file")
        
        job_choice = input("\\nChoice (1 or 2): ").strip()
        
        if job_choice == '1':
            print("\\nEnter the job description (press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done):")
            job_text = sys.stdin.read().strip()
        elif job_choice == '2':
            job_file = input("\\nEnter path to job description file: ").strip().strip('"')
            if not os.path.exists(job_file):
                print(f"\\n❌ Job description file not found: {job_file}")
                return
            with open(job_file, 'r', encoding='utf-8') as f:
                job_text = f.read()
        else:
            print("\\n❌ Invalid choice.")
            return
        
        job_title = input("\\nEnter job title (optional): ").strip() or "Not Specified"
        
        # Perform analysis
        try:
            print("\\n🔄 Analyzing... This may take a few moments.")
            results = self.analyze_single_resume(resume_path, job_text, job_title)
            
            # Display results
            print("\\n" + self.matcher.format_match_results(results))
            
            # Save results option
            save_choice = input("\\n💾 Save results to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"analysis_results_{Path(resume_path).stem}.json"
                self.save_results_to_file(results, filename)
                
        except Exception as e:
            print(f"\\n❌ Analysis failed: {e}")

    def _interactive_batch_analysis(self):
        """Handle batch resume analysis in interactive mode"""
        
        print("\\n" + "-"*40)
        print("📁 Batch Resume Analysis")
        print("-"*40)
        
        # Get resume folder path
        resume_folder = input("\\nEnter the path to folder containing resume PDFs: ").strip().strip('"')
        
        if not os.path.exists(resume_folder):
            print(f"\\n❌ Resume folder not found: {resume_folder}")
            return
        
        # Get job information (same as single analysis)
        print("\\nHow would you like to provide the job description?")
        print("1. Enter job description text directly")
        print("2. Load from a text file")
        
        job_choice = input("\\nChoice (1 or 2): ").strip()
        
        if job_choice == '1':
            print("\\nEnter the job description (press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done):")
            job_text = sys.stdin.read().strip()
        elif job_choice == '2':
            job_file = input("\\nEnter path to job description file: ").strip().strip('"')
            if not os.path.exists(job_file):
                print(f"\\n❌ Job description file not found: {job_file}")
                return
            with open(job_file, 'r', encoding='utf-8') as f:
                job_text = f.read()
        else:
            print("\\n❌ Invalid choice.")
            return
        
        job_title = input("\\nEnter job title (optional): ").strip() or "Not Specified"
        
        # Perform batch analysis
        try:
            print("\\n🔄 Analyzing multiple resumes... This may take several minutes.")
            results = self.analyze_multiple_resumes(resume_folder, job_text, job_title)
            
            if not results:
                print("\\n❌ No resumes were successfully analyzed.")
                return
            
            # Display summary
            print("\\n" + "="*60)
            print("📊 BATCH ANALYSIS RESULTS")
            print("="*60)
            print(f"\\nTotal resumes analyzed: {len(results)}")
            print("\\nTop candidates (sorted by match score):")
            print("-" * 50)
            
            for i, result in enumerate(results[:10], 1):  # Show top 10
                filename = result.get('resume_filename', 'Unknown')
                score = result.get('overall_score', 0)
                category = result.get('match_category', 'Unknown')
                print(f"{i:2d}. {filename:<30} | {score:5.1f}% | {category}")
            
            # Save results option
            save_choice = input("\\n💾 Save detailed results to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"batch_analysis_results_{job_title.replace(' ', '_')}.json"
                self.save_results_to_file(results, filename)
                
        except Exception as e:
            print(f"\\n❌ Batch analysis failed: {e}")


def main():
    """Main entry point of the application"""
    
    parser = argparse.ArgumentParser(
        description="Resume ATS Analyzer - Match resumes against job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py --interactive
  
  # Single resume analysis
  python main.py --resume resume.pdf --job job.txt --title "Software Engineer"
  
  # Batch analysis
  python main.py --batch --resume-folder resumes/ --job job.txt --title "Data Scientist"
        """
    )
    
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    
    parser.add_argument('--resume', type=str,
                       help='Path to resume PDF file')
    
    parser.add_argument('--job', type=str,
                       help='Path to job description text file')
    
    parser.add_argument('--title', type=str, default='Not Specified',
                       help='Job title (optional)')
    
    parser.add_argument('--batch', action='store_true',
                       help='Enable batch processing mode')
    
    parser.add_argument('--resume-folder', type=str,
                       help='Path to folder containing resume PDFs (for batch mode)')
    
    parser.add_argument('--output', type=str, default='analysis_results.json',
                       help='Output file for results (JSON format)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize ATS Analyzer
        analyzer = ATSAnalyzer()
        
        if args.interactive:
            # Interactive mode
            analyzer.interactive_mode()
            
        elif args.batch:
            # Batch mode
            if not args.resume_folder or not args.job:
                print("❌ Error: --resume-folder and --job are required for batch mode")
                sys.exit(1)
            
            if not os.path.exists(args.job):
                print(f"❌ Error: Job description file not found: {args.job}")
                sys.exit(1)
            
            with open(args.job, 'r', encoding='utf-8') as f:
                job_text = f.read()
            
            results = analyzer.analyze_multiple_resumes(
                args.resume_folder, job_text, args.title
            )
            
            if results:
                print(f"\\n✅ Successfully analyzed {len(results)} resumes")
                analyzer.save_results_to_file(results, args.output)
            else:
                print("\\n❌ No resumes were successfully analyzed")
                
        elif args.resume and args.job:
            # Single resume mode
            if not os.path.exists(args.resume):
                print(f"❌ Error: Resume file not found: {args.resume}")
                sys.exit(1)
                
            if not os.path.exists(args.job):
                print(f"❌ Error: Job description file not found: {args.job}")
                sys.exit(1)
            
            with open(args.job, 'r', encoding='utf-8') as f:
                job_text = f.read()
            
            results = analyzer.analyze_single_resume(args.resume, job_text, args.title)
            
            print(analyzer.matcher.format_match_results(results))
            analyzer.save_results_to_file(results, args.output)
            
        else:
            # No valid arguments provided
            print("❌ Error: Invalid arguments. Use --help for usage information.")
            print("\\n💡 Tip: Try running with --interactive for guided setup")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n\\n👋 Operation cancelled by user")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\\n❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
