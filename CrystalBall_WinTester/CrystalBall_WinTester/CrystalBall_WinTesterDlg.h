
// CrystalBall_WinTesterDlg.h : header file
//

#pragma once

#include "../Engine_src/MagicEngine.h"

using namespace CrystalBall;

// CCrystalBall_WinTesterDlg dialog
class CCrystalBall_WinTesterDlg : public CDialogEx
{
// Construction
public:
	CCrystalBall_WinTesterDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CRYSTALBALL_WINTESTER_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	MagicEngine m_MagicEngine;
	afx_msg void OnBnClickedButtonVisionmix();
	afx_msg void OnBnClickedButtonEffect();
};
