// Tab switching
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
        const tabName = button.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        button.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Clear results
        document.getElementById('ingest-result').classList.remove('show');
        document.getElementById('query-result').classList.remove('show');
    });
});

// Ingest form handler
document.getElementById('ingest-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const resultArea = document.getElementById('ingest-result');
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    // Get form data
    const text = document.getElementById('document-text').value.trim();
    const docId = document.getElementById('document-id').value.trim();
    const metadataText = document.getElementById('document-metadata').value.trim();
    
    if (!text) {
        showError(resultArea, 'Document text is required');
        return;
    }
    
    // Parse metadata if provided
    let metadata = null;
    if (metadataText) {
        try {
            metadata = JSON.parse(metadataText);
        } catch (error) {
            showError(resultArea, 'Invalid JSON in metadata field');
            return;
        }
    }
    
    // Prepare request body
    const body = { text };
    if (docId) body.doc_id = docId;
    if (metadata) body.metadata = metadata;
    
    // Show loading state
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="loading"></span>Ingesting...';
    resultArea.classList.remove('show', 'success', 'error');
    
    try {
        const response = await fetch('/api/v1/ingest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showSuccess(resultArea, `
                <h3>✅ Document Ingested Successfully</h3>
                <p><strong>Document ID:</strong> ${data.doc_id}</p>
                <p><strong>Chunks Created:</strong> ${data.chunks_created}</p>
                <p>${data.message}</p>
            `);
            
            // Clear form
            document.getElementById('ingest-form').reset();
        } else {
            showError(resultArea, `Error: ${data.detail || 'Failed to ingest document'}`);
        }
    } catch (error) {
        showError(resultArea, `Network error: ${error.message}`);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
});

// Query form handler
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const resultArea = document.getElementById('query-result');
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    // Get query
    const query = document.getElementById('query-text').value.trim();
    
    if (!query) {
        showError(resultArea, 'Query is required');
        return;
    }
    
    // Get advanced options
    const promptStyle = document.getElementById('prompt-style').value;
    const useAgent = document.getElementById('use-agent').checked;
    const useReasoning = document.getElementById('use-reasoning').checked;
    
    // Build request body
    const requestBody = { query };
    if (promptStyle) requestBody.prompt_style = promptStyle;
    if (useAgent) requestBody.use_agent = true;
    if (useReasoning) requestBody.use_reasoning = true;
    
    // Show loading state
    submitButton.disabled = true;
    const loadingText = useAgent ? 'Agent Thinking...' : useReasoning ? 'Reasoning...' : 'Searching...';
    submitButton.innerHTML = `<span class="loading"></span>${loadingText}`;
    resultArea.classList.remove('show', 'success', 'error');
    
    try {
        const response = await fetch('/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            let html = '';
            
            // Model type badge
            if (data.model_type) {
                const badges = {
                    'standard': '📝 Standard',
                    'reasoning': '🧠 Reasoning',
                    'agentic': '🤖 Agentic'
                };
                html += `<div class="model-badge">${badges[data.model_type] || data.model_type}</div>`;
            }
            
            // Reasoning section
            if (data.reasoning) {
                html += `
                    <div class="reasoning-section">
                        <h3>🧠 Reasoning Steps</h3>
                        <div class="reasoning-content">${formatText(data.reasoning)}</div>
                    </div>
                `;
            }
            
            // Agent trace section
            if (data.agent_trace) {
                html += `
                    <div class="agent-trace-section">
                        <h3>🤖 Agent Execution Trace</h3>
                        <div class="trace-content">
                            <p><strong>Iterations:</strong> ${data.agent_trace.iteration || 0}</p>
                            ${data.agent_trace.thoughts && data.agent_trace.thoughts.length > 0 ? `
                                <div class="thoughts">
                                    <strong>Thoughts:</strong>
                                    <ul>
                                        ${data.agent_trace.thoughts.map(t => `<li>${escapeHtml(t)}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            ${data.agent_trace.actions && data.agent_trace.actions.length > 0 ? `
                                <div class="actions">
                                    <strong>Actions:</strong>
                                    <ul>
                                        ${data.agent_trace.actions.map(a => `<li>Used tool: <code>${escapeHtml(a.tool)}</code></li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
            
            // Answer section
            html += `
                <div class="answer-section">
                    <h3>💡 Answer</h3>
                    <p>${formatText(data.answer)}</p>
                </div>
            `;
            
            // Sources section
            if (data.sources && data.sources.length > 0) {
                html += `
                    <div class="sources-section">
                        <h4>📚 Sources (${data.num_sources})</h4>
                `;
                
                data.sources.forEach((source, index) => {
                    html += `
                        <div class="source-item">
                            <strong>Source ${index + 1}: ${escapeHtml(source.doc_id)}</strong>
                            ${source.chunk_index !== null ? `<span style="color: var(--text-secondary);">Chunk ${source.chunk_index}</span>` : ''}
                            <div class="preview">${escapeHtml(source.text_preview)}</div>
                        </div>
                    `;
                });
                
                html += '</div>';
            } else {
                html += '<p style="color: var(--text-secondary);">No sources found.</p>';
            }
            
            showSuccess(resultArea, html);
        } else {
            showError(resultArea, `Error: ${data.detail || 'Failed to process query'}`);
        }
    } catch (error) {
        showError(resultArea, `Network error: ${error.message}`);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
});

// Helper functions
function showSuccess(element, html) {
    element.innerHTML = html;
    element.classList.add('show', 'success');
    element.classList.remove('error');
    element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(element, message) {
    element.innerHTML = `<h3>❌ Error</h3><p>${escapeHtml(message)}</p>`;
    element.classList.add('show', 'error');
    element.classList.remove('success');
    element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatText(text) {
    // Convert markdown-style formatting to HTML
    if (!text) return '';
    
    // Escape HTML first
    let formatted = escapeHtml(text);
    
    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Convert numbered lists (1. 2. etc.)
    formatted = formatted.replace(/(\d+)\.\s+(.+?)(?=\d+\.|$)/g, '<li>$2</li>');
    if (formatted.includes('<li>')) {
        formatted = '<ol>' + formatted + '</ol>';
    }
    
    return formatted;
}

